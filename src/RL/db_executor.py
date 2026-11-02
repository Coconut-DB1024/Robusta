"""
统一数据库执行器。

职责：
- 只负责数据库连接、SQL 执行、事务和统一结果封装；
- 不负责判断谓词是否正确，也不负责 SQL 方言渲染；
- 通过注册表支持后续继续接入 PostgreSQL 以外的 DBMS。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Type

import pymysql
import pymysql.cursors

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # 允许只跑 MySQL 环境时不安装 psycopg
    psycopg = None
    dict_row = None

# Oracle 驱动为可选依赖；只有选择 Oracle 分支时才要求安装。
try:
    import oracledb
except ImportError:
    oracledb = None

# SQL Server 驱动也是可选依赖，不影响其他数据库的执行器加载。
try:
    import pyodbc
except ImportError:
    pyodbc = None

from RL.config import DB_CONFIG


@dataclass
class DBQueryResult:
    """封装一次 SQL 执行结果，供 RL / BugFinding 复用。"""

    sql: str
    success: bool
    rows: Optional[List[Dict[str, Any]]] = None
    rowcount: int = 0
    error: Optional[str] = None
    error_code: Optional[int] = None
    execution_time: float = 0.0
    warning_count: int = 0
    timed_out: bool = False
    sqlstate: Optional[str] = None
    cancelled: bool = False


class DBExecutorError(Exception):
    """数据库环境级错误，例如连接失败、驱动未安装、数据库不存在。"""

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        original: Optional[BaseException] = None,
        sqlstate: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.original = original
        self.sqlstate = sqlstate

    def __str__(self) -> str:
        base = self.message
        if self.error_code is not None:
            base += f" (error_code={self.error_code})"
        return base


class BaseDBExecutor:
    """
    多 DBMS 执行器基类。

    新增数据库时，只需要继承该类并实现：
    - db_type
    - driver_error
    - _create_connection()
    - _get_autocommit()
    - _set_autocommit()
    - _connection_is_alive()
    - _get_warning_count() 可选
    """

    db_type: str = "base"
    driver_error: Type[BaseException] = Exception

    def __init__(
        self,
        autoconnect: bool = True,
        config: Any = None,
        query_timeout_seconds: Optional[int] = None,
    ) -> None:
        self.config = config or DB_CONFIG
        # None 保持调用方原有行为；第三阶段可显式传入独立的 SQL 超时。
        self.query_timeout_seconds = (
            int(query_timeout_seconds) if query_timeout_seconds is not None else None
        )
        self._conn: Any = None
        if autoconnect:
            self._conn = self._create_connection()

    def _create_connection(self) -> Any:
        raise NotImplementedError

    def _get_autocommit(self) -> bool:
        raise NotImplementedError

    def _set_autocommit(self, enabled: bool) -> None:
        raise NotImplementedError

    def _connection_is_alive(self) -> bool:
        raise NotImplementedError

    def _get_warning_count(self, cursor: Any) -> int:
        return 0

    def _prepare_sql(self, sql: str) -> str:
        """为驱动准备 SQL；默认保持原 SQL，避免改变现有 DBMS 行为。"""
        return sql

    def _configure_cursor(self, cursor: Any) -> None:
        """按驱动需要配置游标；默认不做处理。"""
        return None

    def _extract_error_code(self, exc: BaseException) -> Optional[int]:
        if hasattr(exc, "args") and getattr(exc, "args"):
            try:
                return int(exc.args[0])
            except Exception:
                return None
        return None

    def _extract_error_message(self, exc: BaseException) -> str:
        if hasattr(exc, "args") and len(getattr(exc, "args", ())) >= 2:
            if isinstance(exc.args[1], str):
                return exc.args[1]
        return str(exc)

    def _is_query_timeout(self, exc: BaseException) -> bool:
        """由具体 DBMS 执行器识别其驱动返回的查询超时。"""
        return False

    def _ensure_connection(self) -> None:
        if self._conn is None:
            self._conn = self._create_connection()
            return

        try:
            if self._connection_is_alive():
                return
        except Exception:
            pass

        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = self._create_connection()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def ping(self) -> bool:
        try:
            self._ensure_connection()
            return True
        except Exception:
            return False

    def execute_query(
        self,
        sql: str,
        params: Optional[tuple] = None,
        fetch: str = "all",
        autocommit: bool = True,
    ) -> DBQueryResult:
        self._ensure_connection()
        assert self._conn is not None

        if not autocommit and self._get_autocommit():
            self._set_autocommit(False)

        cursor = self._conn.cursor()
        driver_sql = self._prepare_sql(sql)
        start_time = perf_counter()
        rows: Optional[List[Dict[str, Any]]] = None
        rowcount = 0
        warning_count = 0

        try:
            cursor.execute(driver_sql, params)
            # Oracle 的 rowfactory 必须在 execute() 得到列元数据后设置。
            # 其他执行器未覆盖该钩子，因此原有 MySQL/PostgreSQL 行为不变。
            self._configure_cursor(cursor)
            rowcount = cursor.rowcount

            if fetch == "all":
                fetched = cursor.fetchall()
                rows = list(fetched or [])
            elif fetch == "one":
                one = cursor.fetchone()
                rows = [one] if one is not None else []
            else:
                rows = []

            warning_count = self._get_warning_count(cursor)

            if autocommit and not self._get_autocommit():
                self._conn.commit()
                self._set_autocommit(True)

            return DBQueryResult(
                sql=sql,
                success=True,
                rows=rows,
                rowcount=rowcount,
                error=None,
                error_code=None,
                execution_time=perf_counter() - start_time,
                warning_count=warning_count,
                timed_out=False,
            )

        except self.driver_error as e:
            if not autocommit:
                try:
                    self._conn.rollback()
                except Exception:
                    pass

            return DBQueryResult(
                sql=sql,
                success=False,
                rows=None,
                rowcount=rowcount,
                error=self._extract_error_message(e),
                error_code=self._extract_error_code(e),
                execution_time=perf_counter() - start_time,
                warning_count=warning_count,
                timed_out=self._is_query_timeout(e),
            )

        except Exception as e:
            if not autocommit:
                try:
                    self._conn.rollback()
                except Exception:
                    pass

            return DBQueryResult(
                sql=sql,
                success=False,
                rows=None,
                rowcount=rowcount,
                error=str(e),
                error_code=None,
                execution_time=perf_counter() - start_time,
                warning_count=warning_count,
                timed_out=self._is_query_timeout(e),
            )

        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def execute_queries_in_transaction(self, sql_list: List[str]) -> List[DBQueryResult]:
        self._ensure_connection()
        assert self._conn is not None

        self._set_autocommit(False)
        results: List[DBQueryResult] = []
        all_success = True

        try:
            for sql in sql_list:
                res = self.execute_query(
                    sql=sql,
                    params=None,
                    fetch="none",
                    autocommit=False,
                )
                results.append(res)
                if not res.success:
                    all_success = False
                    break

            if all_success:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            try:
                self._set_autocommit(True)
            except Exception:
                pass

        return results

    def test_simple_select(self) -> DBQueryResult:
        return self.execute_query(
            sql="SELECT 1 AS test_value;",
            params=None,
            fetch="one",
            autocommit=True,
        )

    def list_tables(self) -> DBQueryResult:
        raise NotImplementedError(f"{self.__class__.__name__}.list_tables() is not implemented.")

    def list_columns(self, table_name: str, schema: Optional[str] = None) -> DBQueryResult:
        raise NotImplementedError(f"{self.__class__.__name__}.list_columns() is not implemented.")


class MySQLExecutor(BaseDBExecutor):
    """MySQL / MySQL-like 数据库执行器。保留类名以兼容现有调用点。"""

    db_type = "mysql"
    driver_error = pymysql.MySQLError

    def __init__(
        self,
        autoconnect: bool = True,
        config: Any = None,
        query_timeout_seconds: Optional[int] = None,
    ) -> None:
        self._cursor_cls = pymysql.cursors.DictCursor
        super().__init__(
            autoconnect=autoconnect,
            config=config,
            query_timeout_seconds=query_timeout_seconds,
        )

    def _create_connection(self) -> pymysql.connections.Connection:
        query_timeout = self.query_timeout_seconds
        if query_timeout is None:
            query_timeout = self.config.query_timeout
        elif query_timeout <= 0:
            query_timeout = None

        try:
            return pymysql.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                unix_socket=getattr(self.config, "unix_socket", None),
                connect_timeout=self.config.connect_timeout,
                read_timeout=query_timeout,
                write_timeout=query_timeout,
                charset="utf8mb4",
                cursorclass=self._cursor_cls,
                autocommit=True,
            )
        except pymysql.err.OperationalError as e:
            raise DBExecutorError(
                message=f"Failed to connect to MySQL: {e}",
                error_code=self._extract_error_code(e),
                original=e,
            ) from e

    def _is_query_timeout(self, exc: BaseException) -> bool:
        # PyMySQL 通常把 socket 读取超时包装为 2006/2013 OperationalError。
        message = self._extract_error_message(exc).lower()
        return "timed out" in message or "timeout" in message

    def _get_autocommit(self) -> bool:
        return bool(self._conn.get_autocommit())

    def _set_autocommit(self, enabled: bool) -> None:
        self._conn.autocommit(bool(enabled))

    def _connection_is_alive(self) -> bool:
        self._conn.ping(reconnect=True)
        return True

    def _get_warning_count(self, cursor: Any) -> int:
        try:
            return int(cursor.connection.warning_count())
        except Exception:
            return 0

    def list_tables(self) -> DBQueryResult:
        return self.execute_query("SHOW TABLES;", fetch="all", autocommit=True)

    def list_columns(self, table_name: str, schema: Optional[str] = None) -> DBQueryResult:
        quoted = "`" + str(table_name).replace("`", "``") + "`"
        return self.execute_query(f"SHOW COLUMNS FROM {quoted};", fetch="all", autocommit=True)


class PostgreSQLExecutor(BaseDBExecutor):
    """PostgreSQL 数据库执行器。"""

    db_type = "postgresql"
    driver_error = psycopg.Error if psycopg is not None else Exception

    def _create_connection(self) -> Any:
        if psycopg is None or dict_row is None:
            raise DBExecutorError(
                message="Failed to connect to PostgreSQL: psycopg is not installed.",
                error_code=None,
                original=None,
            )

        connect_kwargs = {
            "host": self.config.host,
            "port": self.config.port,
            "user": self.config.user,
            "password": self.config.password,
            "dbname": self.config.database,
            "connect_timeout": self.config.connect_timeout,
            "row_factory": dict_row,
        }
        if self.query_timeout_seconds is not None and self.query_timeout_seconds > 0:
            timeout_ms = int(self.query_timeout_seconds * 1000)
            connect_kwargs["options"] = f"-c statement_timeout={timeout_ms}"

        try:
            conn = psycopg.connect(
                **connect_kwargs,
            )
            conn.autocommit = True
            return conn
        except psycopg.Error as e:
            raise DBExecutorError(
                message=f"Failed to connect to PostgreSQL: {e}",
                error_code=self._extract_error_code(e),
                original=e,
            ) from e

    def _get_autocommit(self) -> bool:
        return bool(self._conn.autocommit)

    def _set_autocommit(self, enabled: bool) -> None:
        self._conn.autocommit = bool(enabled)

    def _connection_is_alive(self) -> bool:
        if getattr(self._conn, "closed", False):
            return False
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
            return True
        except Exception:
            return False

    def _extract_error_code(self, exc: BaseException) -> Optional[int]:
        sqlstate = getattr(exc, "sqlstate", None)
        if sqlstate is None:
            return None
        try:
            return int(sqlstate)
        except Exception:
            return None

    def _extract_error_message(self, exc: BaseException) -> str:
        diag = getattr(exc, "diag", None)
        primary = getattr(diag, "message_primary", None) if diag is not None else None
        return str(primary or exc)

    def _is_query_timeout(self, exc: BaseException) -> bool:
        sqlstate = str(getattr(exc, "sqlstate", "") or "")
        message = self._extract_error_message(exc).lower()
        return sqlstate == "57014" and "statement timeout" in message

    def list_tables(self) -> DBQueryResult:
        return self.execute_query(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name;
            """,
            fetch="all",
            autocommit=True,
        )

    def list_columns(self, table_name: str, schema: Optional[str] = None) -> DBQueryResult:
        return self.execute_query(
            """
            SELECT column_name AS "Field", data_type AS "Type"
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
            ORDER BY ordinal_position;
            """,
            params=(schema or "public", table_name),
            fetch="all",
            autocommit=True,
        )


class OracleExecutor(BaseDBExecutor):
    """Oracle 执行器（python-oracledb thin/thick 模式均可用）。"""

    db_type = "oracle"
    driver_error = oracledb.Error if oracledb is not None else Exception

    def __init__(
        self,
        autoconnect: bool = True,
        config: Any = None,
        query_timeout_seconds: Optional[int] = None,
    ) -> None:
        self.initialization_error: Optional[DBExecutorError] = None
        try:
            super().__init__(
                autoconnect=autoconnect,
                config=config,
                query_timeout_seconds=query_timeout_seconds,
            )
        except DBExecutorError as exc:
            # Oracle 环境不可用时保留执行器实例，让 candidate_bitmap 记录环境错误，
            # 而不是在构造阶段直接终止 Stage 2。
            self.initialization_error = exc
            self._conn = None

    def _create_connection(self) -> Any:
        if oracledb is None:
            raise DBExecutorError(
                message="Failed to connect to Oracle: python-oracledb is not installed.",
                error_code=None,
                original=None,
            )

        host = str(getattr(self.config, "host", "127.0.0.1"))
        try:
            port = int(getattr(self.config, "port", 1521))
        except (TypeError, ValueError) as exc:
            raise DBExecutorError(
                message=f"Failed to connect to Oracle: invalid port {getattr(self.config, 'port', None)!r}.",
                error_code=None,
                original=exc,
            ) from exc
        service_name = str(getattr(self.config, "database", "") or "").strip()
        if not service_name:
            raise DBExecutorError(
                message="Failed to connect to Oracle: database must contain the service name.",
                error_code=None,
                original=None,
            )
        try:
            # 本项目沿用 database 字段存放 Oracle service name。
            dsn = oracledb.makedsn(host, port, service_name=service_name)
        except Exception as exc:
            raise DBExecutorError(
                message=f"Failed to build Oracle DSN: {exc}",
                error_code=self._extract_error_code(exc),
                original=exc,
            ) from exc

        connect_kwargs: Dict[str, Any] = {
            "user": getattr(self.config, "user", ""),
            "password": getattr(self.config, "password", ""),
            "dsn": dsn,
        }
        connect_timeout = int(getattr(self.config, "connect_timeout", 0) or 0)
        if connect_timeout > 0:
            connect_kwargs["tcp_connect_timeout"] = connect_timeout

        try:
            conn = oracledb.connect(**connect_kwargs)
            query_timeout = self.query_timeout_seconds
            if query_timeout is None:
                query_timeout = int(getattr(self.config, "query_timeout", 0) or 0)
            if query_timeout and query_timeout > 0:
                # python-oracledb 的 call_timeout 单位为毫秒。
                try:
                    conn.call_timeout = int(query_timeout * 1000)
                except (AttributeError, TypeError):
                    # 老版本/测试桩不支持 call_timeout 时仍可建立连接。
                    pass
            conn.autocommit = True
            return conn
        except DBExecutorError:
            raise
        except Exception as exc:
            raise DBExecutorError(
                message=f"Failed to connect to Oracle: {self._extract_error_message(exc)}",
                error_code=self._extract_error_code(exc),
                original=exc,
            ) from exc

    def _prepare_sql(self, sql: str) -> str:
        # Oracle 驱动执行单条语句时不接受尾部分号；返回值只供驱动使用。
        return str(sql).rstrip().rstrip(";").rstrip()

    def _configure_cursor(self, cursor: Any) -> None:
        # 项目上层统一按 dict 读取 Schema；Oracle 默认 fetch 返回 tuple，故在此适配。
        # 此方法由基类在 execute() 之后调用，此时 cursor.description 已可用。
        description = getattr(cursor, "description", None) or []
        column_names = []
        for item in description:
            name = getattr(item, "name", None)
            if name is None:
                try:
                    name = item[0]
                except (TypeError, IndexError):
                    name = str(item)
            column_names.append(str(name))

        def rowfactory(*values: Any) -> Dict[str, Any]:
            return {
                name: value
                for name, value in zip(column_names, values)
            }

        try:
            cursor.rowfactory = rowfactory
        except (AttributeError, TypeError):
            # 仅用于极简测试桩；真实 python-oracledb 支持 rowfactory。
            pass

    def _get_autocommit(self) -> bool:
        return bool(getattr(self._conn, "autocommit", True))

    def _set_autocommit(self, enabled: bool) -> None:
        self._conn.autocommit = bool(enabled)

    def _connection_is_alive(self) -> bool:
        if getattr(self._conn, "closed", False):
            return False
        self._conn.ping()
        return True

    def _extract_error_code(self, exc: BaseException) -> Optional[int]:
        # python-oracledb 通常把 error 对象放在 args[0]，其 code 为正整数。
        args = getattr(exc, "args", ()) or ()
        for item in args:
            value = getattr(item, "code", None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        message = self._extract_error_message(exc)
        import re

        match = re.search(r"(?:ORA|DPY)-(\d+)", message, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None

    def _extract_error_message(self, exc: BaseException) -> str:
        args = getattr(exc, "args", ()) or ()
        for item in args:
            message = getattr(item, "message", None)
            if message:
                return str(message)
        return str(exc)

    def _is_query_timeout(self, exc: BaseException) -> bool:
        code = self._extract_error_code(exc)
        text = self._extract_error_message(exc).lower()
        return code in {1013, 51, 4024, 6005} or any(
            marker in text
            for marker in (
                "call timeout",
                "statement timeout",
                "user requested cancel",
                "timed out",
                "timeout",
            )
        )

    def execute_query(
        self,
        sql: str,
        params: Optional[tuple] = None,
        fetch: str = "all",
        autocommit: bool = True,
    ) -> DBQueryResult:
        try:
            return super().execute_query(
                sql=sql,
                params=params,
                fetch=fetch,
                autocommit=autocommit,
            )
        except DBExecutorError as exc:
            # _ensure_connection 在真正执行前失败时，统一转换为查询结果。
            return DBQueryResult(
                sql=sql,
                success=False,
                rows=None,
                rowcount=0,
                error=self._extract_error_message(exc),
                error_code=exc.error_code,
                execution_time=0.0,
                warning_count=0,
                timed_out=False,
            )

    def test_simple_select(self) -> DBQueryResult:
        return self.execute_query(
            sql="SELECT 1 AS test_value FROM dual",
            params=None,
            fetch="one",
            autocommit=True,
        )

    def list_tables(self) -> DBQueryResult:
        return self.execute_query(
            'SELECT table_name AS "table_name" FROM user_tables ORDER BY table_name',
            fetch="all",
            autocommit=True,
        )

    def list_columns(self, table_name: str, schema: Optional[str] = None) -> DBQueryResult:
        if schema:
            return self.execute_query(
                """
                SELECT column_name AS "Field", data_type AS "Type"
                FROM all_tab_columns
                WHERE owner = UPPER(:1) AND table_name = UPPER(:2)
                ORDER BY column_id
                """,
                params=(schema, table_name),
                fetch="all",
                autocommit=True,
            )
        return self.execute_query(
            """
            SELECT column_name AS "Field", data_type AS "Type"
            FROM user_tab_columns
            WHERE table_name = UPPER(:1)
            ORDER BY column_id
            """,
            params=(table_name,),
            fetch="all",
            autocommit=True,
        )


class SQLServerExecutor(BaseDBExecutor):
    """SQL Server 2022 执行器，使用 pyodbc / Microsoft ODBC Driver 18。"""

    db_type = "sqlserver"
    driver_error = pyodbc.Error if pyodbc is not None else Exception

    def __init__(
        self,
        autoconnect: bool = True,
        config: Any = None,
        query_timeout_seconds: Optional[int] = None,
    ) -> None:
        self.initialization_error: Optional[DBExecutorError] = None
        self._manual_transaction = False
        try:
            super().__init__(
                autoconnect=autoconnect,
                config=config,
                query_timeout_seconds=query_timeout_seconds,
            )
        except DBExecutorError as exc:
            # 与 Oracle 一致：环境不可用时，后续查询仍能返回统一的错误结果。
            self.initialization_error = exc
            self._conn = None

    @staticmethod
    def _odbc_value(value: Any) -> str:
        # ODBC 的花括号转义；口令中的分号、右花括号不能变成连接选项。
        return "{" + str(value).replace("}", "}}") + "}"

    def _connection_flag(self, name: str, default: bool) -> str:
        value = getattr(self.config, name, default)
        text = str(value).strip().lower()
        if text in {"true", "yes", "1"}:
            return "yes"
        if text in {"false", "no", "0"}:
            return "no"
        raise ValueError(f"Invalid SQL Server connection option: {name}")

    @staticmethod
    def _decode_datetimeoffset(raw: bytes) -> Any:
        # pyodbc 5.3 不内置 SQL_SS_TIMESTAMPOFFSET (-155) 转换。
        # 对齐其 datetime2 返回值的 Python 微秒精度，同时保留时区偏移。
        from datetime import datetime, timedelta, timezone
        import struct

        year, month, day, hour, minute, second, fraction, tz_hour, tz_minute = struct.unpack(
            "<6hI2h", raw
        )
        return datetime(
            year, month, day, hour, minute, second, fraction // 1000,
            tzinfo=timezone(timedelta(hours=tz_hour, minutes=tz_minute)),
        )

    def _create_connection(self) -> Any:
        if pyodbc is None:
            raise DBExecutorError("Failed to connect to SQL Server: pyodbc is not installed.")

        conn = None
        try:
            host = str(getattr(self.config, "host", "127.0.0.1") or "").strip()
            port = int(getattr(self.config, "port", 1433))
            database = str(getattr(self.config, "database", "") or "").strip()
            if not host or not database or not 1 <= port <= 65535:
                raise ValueError("SQL Server requires host, database and a TCP port in 1..65535.")
            driver = str(
                getattr(self.config, "odbc_driver", "ODBC Driver 18 for SQL Server")
            ).strip()
            if not driver:
                raise ValueError("SQL Server ODBC driver name must not be empty.")
            options = [
                "DRIVER=" + self._odbc_value(driver),
                "SERVER=" + self._odbc_value(f"tcp:{host},{port}"),
                "DATABASE=" + self._odbc_value(database),
                "Encrypt=" + self._connection_flag("encrypt", True),
                # 对齐本项目独立 SQL Server 初始化脚本的本地实验连接设置。
                # 可由传入 config.trust_server_certificate=False 启用证书验证。
                "TrustServerCertificate=" + self._connection_flag("trust_server_certificate", True),
            ]
            if self._connection_flag("trusted_connection", False) == "yes":
                options.append("Trusted_Connection=yes")
            else:
                options.extend([
                    "UID=" + self._odbc_value(getattr(self.config, "user", "")),
                    "PWD=" + self._odbc_value(getattr(self.config, "password", "")),
                ])

            connect_timeout = max(0, int(getattr(self.config, "connect_timeout", 5)))
            query_timeout = self.query_timeout_seconds
            if query_timeout is None:
                query_timeout = int(getattr(self.config, "query_timeout", 10))
            conn = pyodbc.connect(
                ";".join(options) + ";", timeout=connect_timeout, autocommit=True
            )
            # connect(timeout=...) 只控制登录；Connection.timeout 对新游标设置
            # SQL_ATTR_QUERY_TIMEOUT（秒）。不能用登录超时代替查询超时。
            conn.timeout = max(0, int(query_timeout))
            conn.add_output_converter(-155, self._decode_datetimeoffset)
            self._manual_transaction = False
            return conn
        except Exception as exc:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            raise DBExecutorError(
                message=f"Failed to connect to SQL Server: {self._extract_error_message(exc)}",
                error_code=self._extract_error_code(exc),
                original=exc,
                sqlstate=self._extract_sqlstate(exc),
            ) from exc

    def _get_autocommit(self) -> bool:
        return bool(self._conn.autocommit)

    def _set_autocommit(self, enabled: bool) -> None:
        self._conn.autocommit = bool(enabled)
        self._manual_transaction = not bool(enabled)

    def _ensure_connection(self) -> None:
        # 事务内不得通过 ping/reconnect 偷换会话，否则可能只提交事务的后半段。
        # 坏连接由当前语句报错，事务调用方负责回滚。
        if self._conn is not None and self._manual_transaction:
            return
        super()._ensure_connection()

    def _connection_is_alive(self) -> bool:
        cursor = self._conn.cursor()
        try:
            cursor.execute("SELECT 1")
            return cursor.fetchone() is not None
        finally:
            cursor.close()

    @staticmethod
    def _extract_sqlstate(exc: BaseException) -> Optional[str]:
        state = getattr(exc, "sqlstate", None)
        if state:
            return str(state)
        args = getattr(exc, "args", ()) or ()
        if args and re.fullmatch(r"[A-Z0-9]{5}", str(args[0])):
            return str(args[0])
        for arg in args:
            match = re.search(r"\[([A-Z0-9]{5})\]", str(arg))
            if match:
                return match.group(1)
        return None

    def _extract_error_code(self, exc: BaseException) -> Optional[int]:
        if isinstance(exc, DBExecutorError):
            return exc.error_code
        # pyodbc 将原生错误码放在诊断文本尾部；不要把 SQLSTATE 当整数错误码。
        codes = re.findall(
            r"\((-?\d+)\)(?=\s*(?:\([A-Za-z][A-Za-z0-9_]*\)|;|$))",
            self._extract_error_message(exc),
        )
        return next((int(code) for code in codes if int(code) != 0), int(codes[0]) if codes else None)

    def _extract_error_message(self, exc: BaseException) -> str:
        if isinstance(exc, DBExecutorError):
            return exc.message
        args = getattr(exc, "args", ()) or ()
        return " | ".join(str(arg) for arg in args) if args else str(exc)

    def _is_query_timeout(self, exc: BaseException) -> bool:
        return self._extract_sqlstate(exc) in {"HYT00", "HYT01", "S1T00"} or self._extract_error_code(exc) == -2

    def _rollback_after_error(self) -> None:
        if self._conn is not None:
            try:
                if not self._get_autocommit():
                    self._conn.rollback()
            except Exception:
                # 回滚失败时丢弃会话，避免下一次调用提交不确定的事务。
                try:
                    self.close()
                except Exception:
                    pass

    def execute_query(
        self,
        sql: str,
        params: Optional[tuple] = None,
        fetch: str = "all",
        autocommit: bool = True,
    ) -> DBQueryResult:
        # pyodbc 无 rowfactory，且无结果集的 DML 不能 fetchall。
        # 将差异留在子类，保持 MySQL/PostgreSQL/Oracle 的公共路径不变。
        start_time = perf_counter()
        cursor = None
        rowcount = 0
        warning_count = 0
        try:
            self._ensure_connection()
            if not autocommit and self._get_autocommit():
                self._set_autocommit(False)
            cursor = self._conn.cursor()
            if params is None:
                cursor.execute(sql)
            else:
                # 原生 qmark 参数绑定；不改写字符串字面量中的 ? 或其他方言占位符。
                cursor.execute(sql, params)

            rows: List[Dict[str, Any]] = []
            found_result = False
            rowcount = cursor.rowcount
            while True:
                description = cursor.description
                if description is not None and not found_result:
                    found_result = True
                    names = [str(column[0]) for column in description]
                    if fetch == "all":
                        rows = [dict(zip(names, row)) for row in cursor.fetchall()]
                    elif fetch == "one":
                        row = cursor.fetchone()
                        rows = [dict(zip(names, row))] if row is not None else []
                    rowcount = cursor.rowcount
                warning_count += len(getattr(cursor, "messages", None) or [])
                # 跳过 UPDATE 的计数结果，并检查后续结果是否报错。
                # 接口仍只返回第一个表格结果，不增加多结果集 API。
                if not cursor.nextset():
                    break
            if autocommit and not self._get_autocommit():
                self._conn.commit()
                self._set_autocommit(True)
            return DBQueryResult(
                sql=sql, success=True, rows=rows, rowcount=rowcount,
                execution_time=perf_counter() - start_time,
                warning_count=warning_count,
            )
        except Exception as exc:
            self._rollback_after_error()
            # 登录超时属于连接错误，不误报为已执行查询的超时。
            connection_error = isinstance(exc, DBExecutorError)
            state = self._extract_sqlstate(exc)
            return DBQueryResult(
                sql=sql, success=False, rows=None, rowcount=rowcount,
                error=self._extract_error_message(exc),
                error_code=self._extract_error_code(exc),
                execution_time=perf_counter() - start_time,
                warning_count=warning_count,
                timed_out=not connection_error and self._is_query_timeout(exc),
                sqlstate=state,
                cancelled=not connection_error and state == "HY008",
            )
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass

    def execute_queries_in_transaction(self, sql_list: List[str]) -> List[DBQueryResult]:
        self._ensure_connection()
        self._set_autocommit(False)
        results: List[DBQueryResult] = []
        try:
            for sql in sql_list:
                result = self.execute_query(sql, fetch="none", autocommit=False)
                results.append(result)
                if not result.success:
                    return results  # execute_query 已回滚；回滚失败时已关闭连接。
            self._conn.commit()
            return results
        except Exception:
            self._rollback_after_error()
            raise
        finally:
            if self._conn is not None:
                self._set_autocommit(True)

    def _default_schema(self) -> str:
        return str(getattr(self.config, "schema", "dbo") or "dbo")

    def list_tables(self) -> DBQueryResult:
        # 现有启发式只传递 table_name；限定同一个 schema 避免跨 schema 同名表误选。
        return self.execute_query(
            """
            SELECT s.name AS table_schema, t.name AS table_name
            FROM sys.tables AS t
            JOIN sys.schemas AS s ON s.schema_id = t.schema_id
            WHERE t.is_ms_shipped = 0 AND s.name = ?
            ORDER BY t.name;
            """,
            params=(self._default_schema(),),
        )

    def list_columns(self, table_name: str, schema: Optional[str] = None) -> DBQueryResult:
        # 支持 table、schema.table、[schema].[table]；值始终通过 ? 绑定。
        part = r"(?:\[(?:[^\]]|\]\])+\]|[^.\[\]]+)"
        match = re.fullmatch(rf"\s*({part})\s*(?:\.\s*({part}))?\s*", str(table_name))
        if match is None:
            raise ValueError("Expected a SQL Server table name or schema.table.")
        parts = [p.strip() for p in match.groups() if p is not None]
        parts = [p[1:-1].replace("]]", "]") if p.startswith("[") else p for p in parts]
        if len(parts) == 2:
            if schema is not None and schema != parts[0]:
                raise ValueError("Conflicting SQL Server schema arguments.")
            schema, table_name = parts
        else:
            table_name = parts[0]
        return self.execute_query(
            """
            SELECT c.name AS [Field],
                   CASE
                     WHEN ty.is_user_defined = 1
                       THEN QUOTENAME(SCHEMA_NAME(ty.schema_id)) + '.' + QUOTENAME(ty.name)
                     WHEN ty.name IN ('varchar','char','varbinary','binary','nvarchar','nchar')
                       THEN ty.name + '(' +
                         CASE WHEN c.max_length = -1 THEN 'max'
                              ELSE CONVERT(varchar(10), CASE WHEN ty.name IN ('nvarchar','nchar')
                                           THEN c.max_length / 2 ELSE c.max_length END) END + ')'
                     WHEN ty.name IN ('decimal','numeric')
                       THEN ty.name + '(' + CONVERT(varchar(10), c.precision) + ',' +
                            CONVERT(varchar(10), c.scale) + ')'
                     WHEN ty.name IN ('datetime2','datetimeoffset','time')
                       THEN ty.name + '(' + CONVERT(varchar(10), c.scale) + ')'
                     WHEN ty.name = 'float'
                       THEN ty.name + '(' + CONVERT(varchar(10), c.precision) + ')'
                     ELSE ty.name
                   END AS [Type],
                   s.name AS table_schema, t.name AS table_name,
                   ty.name AS data_type, COALESCE(bt.name, ty.name) AS base_type,
                   SCHEMA_NAME(ty.schema_id) AS type_schema, ty.is_user_defined,
                   c.column_id, c.is_nullable,
                   CASE WHEN c.is_nullable = 1 THEN 'YES' ELSE 'NO' END AS [Null],
                   c.max_length,
                   CASE WHEN ty.name IN ('nvarchar','nchar') AND c.max_length <> -1
                          THEN c.max_length / 2
                        WHEN ty.name IN ('varchar','char','nvarchar','nchar') THEN c.max_length
                        ELSE NULL END AS character_maximum_length,
                   c.precision AS [precision], c.scale AS scale,
                   c.is_identity, c.is_computed, c.generated_always_type, c.is_hidden,
                   CAST(CASE WHEN EXISTS (
                       SELECT 1 FROM sys.check_constraints AS ck
                       WHERE ck.parent_object_id = c.object_id AND ck.is_disabled = 0
                         AND (ck.parent_column_id = c.column_id OR EXISTS (
                             SELECT 1 FROM sys.sql_expression_dependencies AS dep
                             WHERE dep.referencing_id = ck.object_id
                               AND dep.referenced_id = c.object_id
                               AND dep.referenced_minor_id = c.column_id
                         ))
                   ) THEN 1 ELSE 0 END AS bit) AS has_check_constraint,
                   CAST(CASE WHEN c.is_identity = 1 OR c.is_computed = 1
                                   OR c.system_type_id = 189
                                   OR c.generated_always_type <> 0 OR c.is_hidden = 1
                             THEN 0 ELSE 1 END AS bit) AS is_updatable
            FROM sys.tables AS t
            JOIN sys.schemas AS s ON s.schema_id = t.schema_id
            JOIN sys.columns AS c ON c.object_id = t.object_id
            JOIN sys.types AS ty ON ty.user_type_id = c.user_type_id
            LEFT JOIN sys.types AS bt
              ON bt.user_type_id = c.system_type_id AND bt.user_type_id = bt.system_type_id
            WHERE s.name = ? AND t.name = ?
            ORDER BY c.column_id;
            """,
            params=(schema if schema is not None else self._default_schema(), table_name),
        )


DB_EXECUTOR_REGISTRY: Dict[str, Type[BaseDBExecutor]] = {}


def register_db_executor(*db_types: str) -> Callable[[Type[BaseDBExecutor]], Type[BaseDBExecutor]]:
    """注册一个 DB executor。新增 DBMS 时用该装饰器挂到工厂上。"""

    def decorator(cls: Type[BaseDBExecutor]) -> Type[BaseDBExecutor]:
        for db_type in db_types:
            normalized = str(db_type).lower().strip()
            if normalized:
                DB_EXECUTOR_REGISTRY[normalized] = cls
        return cls

    return decorator


register_db_executor(
    "mysql",
    "tidb",
    "oceanbase",
    "oceanbase_mysql",
    "mysql_like",
)(MySQLExecutor)
register_db_executor("postgresql", "postgres", "pg")(PostgreSQLExecutor)
register_db_executor("oracle", "oracledb", "oracle_ai26")(OracleExecutor)
register_db_executor("sqlserver")(SQLServerExecutor)


def normalize_db_type(db_type: Optional[str] = None) -> str:
    return str(db_type or getattr(DB_CONFIG, "db_type", "mysql")).lower().strip()


def get_db_executor_class(db_type: Optional[str] = None) -> Type[BaseDBExecutor]:
    normalized = normalize_db_type(db_type)
    try:
        return DB_EXECUTOR_REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(DB_EXECUTOR_REGISTRY))
        raise DBExecutorError(
            message=f"Unsupported db_type={normalized!r}. Registered DB executors: {supported}",
            error_code=None,
            original=exc,
        ) from exc


def create_db_executor(
    db_type: Optional[str] = None,
    autoconnect: bool = True,
    config: Any = None,
    query_timeout_seconds: Optional[int] = None,
) -> BaseDBExecutor:
    executor_cls = get_db_executor_class(db_type)
    return executor_cls(
        autoconnect=autoconnect,
        config=config,
        query_timeout_seconds=query_timeout_seconds,
    )


# 兼容一些可能按旧命名导入的地方。
DBExecutor = BaseDBExecutor


if __name__ == "__main__":
    print("=== DB Executor SelfTest ===")
    print(f"[SelfTest] db_type: {normalize_db_type()}")

    executor = create_db_executor(autoconnect=True)

    print("[SelfTest] ping():", executor.ping())

    res = executor.test_simple_select()
    print("[SelfTest] SELECT 1 success:", res.success)
    print("[SelfTest] result rows:", res.rows)
    print("[SelfTest] execution_time: %.6f s" % res.execution_time)

    res2 = executor.list_tables()
    print("[SelfTest] list_tables success:", res2.success)
    if res2.success:
        print("[SelfTest] tables:", res2.rows)
    else:
        print("[SelfTest] list_tables error:", res2.error)

    executor.close()
    print("=== DB Executor SelfTest Done ===")
