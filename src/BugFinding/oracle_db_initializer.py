"""
Oracle configured-table initializer

目标数据库：
Oracle AI Database 26ai Free

用途：
创建 RL/config 中配置的永久目标表，
填充 500 行具有 NULL、随机值、边界值的数据，
供后续数据库谓词测试使用。

注意：
本脚本设计为只运行一次。
不会 DROP TABLE，也不会清空已有数据。
"""

import json
import os
import random
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path

import oracledb


# ============================================================
# Oracle 连接配置
# ============================================================

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from RL.config import DB_CONFIG, get_target_table


# ============================================================
# 表配置
# ============================================================

TABLE_NAME = DB_CONFIG.table_name
TABLE_SQL = get_target_table("oracle", DB_CONFIG)[2]

# 与原 MySQL 版本保持一致
INITIAL_ROWS = 500

# 与原 MySQL 版本保持一致：15% NULL
NULL_PROBABILITY = 0.15

# 每 100 行提交一次
COMMIT_EVERY = 100


# ============================================================
# 通用随机值生成
# ============================================================

def maybe_null(generator):
    """
    按 NULL_PROBABILITY 概率返回 NULL，
    否则调用 generator 产生正常值。
    """
    if random.random() < NULL_PROBABILITY:
        return None

    return generator()


# ============================================================
# 数值类型
# ============================================================

def random_int():
    """
    普通整数。
    保留原 MySQL -10000 ~ 10000 的主要范围，
    同时提高 0、±1、边界值的出现概率。
    """

    special_values = [
        -10000,
        -1,
        0,
        1,
        10000,
    ]

    if random.random() < 0.20:
        return random.choice(special_values)

    return random.randint(-10000, 10000)


def random_big_int():
    """
    用于 NUMBER(38)，增加较大整数覆盖。
    """

    special_values = [
        -(10 ** 30),
        -(10 ** 18),
        -1,
        0,
        1,
        10 ** 18,
        10 ** 30,
    ]

    if random.random() < 0.20:
        return random.choice(special_values)

    return random.randint(-(10 ** 24), 10 ** 24)


def random_decimal():
    """
    用于 NUMBER(p,s)。
    """

    special_values = [
        -10000.0,
        -1.0,
        0.0,
        1.0,
        10000.0,
    ]

    if random.random() < 0.20:
        return random.choice(special_values)

    return round(
        random.uniform(-10000.0, 10000.0),
        6
    )


def random_float():
    """
    用于 FLOAT / BINARY_FLOAT / BINARY_DOUBLE。
    """

    special_values = [
        -10000.0,
        -1.0,
        -0.0,
        0.0,
        1.0,
        10000.0,
    ]

    if random.random() < 0.20:
        return random.choice(special_values)

    return random.uniform(
        -10000.0,
        10000.0
    )


# ============================================================
# 字符串类型
# ============================================================

def random_ascii_string(max_len=50):
    """
    普通 ASCII 字符串。

    bind variable 会负责转义，
    所以字符串里面可以安全包含单引号。
    """

    length = random.randint(
        1,
        max_len
    )

    chars = (
        string.ascii_letters
        + string.digits
        + " _-@'.,:/"
    )

    return "".join(
        random.choice(chars)
        for _ in range(length)
    )


def random_unicode_string(max_len=50):
    """
    用于 NCHAR / NVARCHAR2 / NCLOB，
    增加中文和 Unicode 数据。
    """

    special_values = [
        "测试",
        "数据库",
        "Oracle",
        "中文English混合",
        "空 格",
        "符号_-=+@#",
        "123456",
        "AaBbCc",
        "数据测试",
    ]

    if random.random() < 0.35:
        return random.choice(special_values)

    return random_ascii_string(max_len)


def random_lob_text():
    """
    生成较长文本，
    用于 CLOB / NCLOB。
    """

    chunks = [
        random_unicode_string(80)
        for _ in range(
            random.randint(4, 15)
        )
    ]

    return " | ".join(chunks)


# ============================================================
# 日期时间类型
# ============================================================

def random_datetime():
    """
    与原 MySQL 版本保持接近：
    当前时间向前约 10 年范围。
    """

    start = (
        datetime.now()
        - timedelta(days=365 * 10)
    )

    end = datetime.now()

    return (
        start
        + (end - start) * random.random()
    )


def random_timestamp_tz_text():
    """
    TIMESTAMP WITH TIME ZONE 使用。

    生成不同 UTC offset，
    用于覆盖时区行为。
    """

    dt = random_datetime()

    offset_hours = random.choice([
        -8,
        -5,
        0,
        1,
        5,
        8,
        9,
    ])

    sign = (
        "+"
        if offset_hours >= 0
        else "-"
    )

    return (
        f"{dt.strftime('%Y-%m-%d %H:%M:%S.%f')} "
        f"{sign}{abs(offset_hours):02d}:00"
    )


# ============================================================
# INTERVAL
# ============================================================

def random_interval_ym():
    """
    INTERVAL YEAR TO MONTH。
    """

    years = random.randint(0, 20)

    months = random.randint(0, 11)

    sign = random.choice([
        "",
        "-"
    ])

    return (
        f"{sign}{years}-{months}"
    )


def random_interval_ds():
    """
    INTERVAL DAY TO SECOND。
    """

    days = random.randint(0, 300)

    hours = random.randint(0, 23)

    minutes = random.randint(0, 59)

    seconds = random.randint(0, 59)

    micros = random.randint(
        0,
        999999
    )

    sign = random.choice([
        "",
        "-"
    ])

    return (
        f"{sign}{days} "
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}."
        f"{micros:06d}"
    )


# ============================================================
# 二进制类型
# ============================================================

def random_raw(max_len=32):
    """
    RAW。
    """
    return os.urandom(
        random.randint(
            1,
            max_len
        )
    )


def random_blob(max_len=256):
    """
    BLOB。
    """
    return os.urandom(
        random.randint(
            1,
            max_len
        )
    )


# ============================================================
# BOOLEAN
# ============================================================

def random_boolean_text():
    """
    使用字符串，
    后面交给 Oracle TO_BOOLEAN 转换。
    """

    return random.choice([
        "TRUE",
        "FALSE",
    ])


# ============================================================
# JSON
# ============================================================

def random_json_text():
    """
    Oracle 原生 JSON 类型。
    """

    obj = {
        "id": random.randint(1, 100),

        "tag": "".join(
            random.choices(
                string.ascii_lowercase,
                k=5
            )
        ),

        "score": round(
            random.random(),
            4
        ),

        "active": random.choice([
            True,
            False
        ]),

        "note": random_unicode_string(20),

        "items": [
            random.randint(-5, 5)
            for _ in range(
                random.randint(0, 4)
            )
        ],
    }

    return json.dumps(
        obj,
        ensure_ascii=False
    )


# ============================================================
# XMLTYPE
# ============================================================

def random_xml_text():

    row_id = random.randint(
        1,
        10000
    )

    score = round(
        random.random(),
        4
    )

    tag = "".join(
        random.choices(
            string.ascii_letters,
            k=8
        )
    )

    return (
        "<row>"
        f"<id>{row_id}</id>"
        f"<tag>{tag}</tag>"
        f"<score>{score}</score>"
        "</row>"
    )


# ============================================================
# VECTOR
# ============================================================

def random_vector_text():
    """
    Oracle 26ai VECTOR。
    使用 4 维 FLOAT32。
    """

    values = [
        round(
            random.uniform(
                -10.0,
                10.0
            ),
            6
        )
        for _ in range(4)
    ]

    return (
        "["
        + ",".join(
            str(v)
            for v in values
        )
        + "]"
    )


# ============================================================
# 建表 SQL
# ============================================================

CREATE_TABLE_SQL = f"""
CREATE TABLE {TABLE_SQL} (

    id NUMBER
       GENERATED BY DEFAULT AS IDENTITY
       PRIMARY KEY,

    -- ================= 数值 =================

    col_integer_0 NUMBER(10),

    col_integer_1 INTEGER,

    col_integer_2 NUMBER(38),

    col_decimal_0 NUMBER(18,6),

    col_float_0 FLOAT(126),

    col_float_1 BINARY_FLOAT,

    col_float_2 BINARY_DOUBLE,


    -- ================= 字符串 =================

    col_string_0 VARCHAR2(255),

    col_string_1 CHAR(50),

    col_string_2 NVARCHAR2(255),

    col_string_3 NCHAR(50),

    col_string_4 CLOB,

    col_string_5 NCLOB,


    -- ================= 日期时间 =================

    col_datetime_0 DATE,

    col_datetime_1 TIMESTAMP(6),

    col_datetime_2
        TIMESTAMP(6) WITH TIME ZONE,

    col_datetime_3
        TIMESTAMP(6) WITH LOCAL TIME ZONE,


    -- ================= INTERVAL =================

    col_interval_0
        INTERVAL YEAR(4) TO MONTH,

    col_interval_1
        INTERVAL DAY(3) TO SECOND(6),


    -- ================= 二进制 =================

    col_binary_0 RAW(64),

    col_binary_1 BLOB,


    -- ================= Oracle 其他类型 =================

    col_boolean_0 BOOLEAN,

    col_json_0 JSON,

    col_xml_0 XMLTYPE,

    col_vector_0 VECTOR(4, FLOAT32)
)
"""


# ============================================================
# INSERT SQL
# ============================================================

INSERT_SQL = f"""
INSERT INTO {TABLE_SQL} (

    col_integer_0,
    col_integer_1,
    col_integer_2,

    col_decimal_0,
    col_float_0,
    col_float_1,
    col_float_2,

    col_string_0,
    col_string_1,
    col_string_2,
    col_string_3,
    col_string_4,
    col_string_5,

    col_datetime_0,
    col_datetime_1,
    col_datetime_2,
    col_datetime_3,

    col_interval_0,
    col_interval_1,

    col_binary_0,
    col_binary_1,

    col_boolean_0,
    col_json_0,
    col_xml_0,
    col_vector_0

) VALUES (

    :col_integer_0,
    :col_integer_1,
    :col_integer_2,

    :col_decimal_0,
    :col_float_0,
    :col_float_1,
    :col_float_2,

    :col_string_0,
    :col_string_1,
    :col_string_2,
    :col_string_3,
    :col_string_4,
    :col_string_5,

    :col_datetime_0,
    :col_datetime_1,

    TO_TIMESTAMP_TZ(
        :col_datetime_2,
        'YYYY-MM-DD HH24:MI:SS.FF TZH:TZM'
    ),

    TO_TIMESTAMP_TZ(
        :col_datetime_3,
        'YYYY-MM-DD HH24:MI:SS.FF TZH:TZM'
    ),

    TO_YMINTERVAL(
        :col_interval_0
    ),

    TO_DSINTERVAL(
        :col_interval_1
    ),

    :col_binary_0,
    :col_binary_1,

    TO_BOOLEAN(
        :col_boolean_0
    ),

    JSON(
        :col_json_0
    ),

    XMLTYPE(
        :col_xml_0
    ),

    TO_VECTOR(
        :col_vector_0,
        4,
        FLOAT32
    )
)
"""


# ============================================================
# 生成一行数据
# ============================================================

def make_row():

    return {

        # INTEGER
        "col_integer_0":
            maybe_null(random_int),

        "col_integer_1":
            maybe_null(random_int),

        "col_integer_2":
            maybe_null(random_big_int),


        # FLOAT / NUMBER
        "col_decimal_0":
            maybe_null(random_decimal),

        "col_float_0":
            maybe_null(random_float),

        "col_float_1":
            maybe_null(random_float),

        "col_float_2":
            maybe_null(random_float),


        # STRING
        "col_string_0":
            maybe_null(
                lambda:
                random_ascii_string(80)
            ),

        "col_string_1":
            maybe_null(
                lambda:
                random_ascii_string(40)
            ),

        "col_string_2":
            maybe_null(
                lambda:
                random_unicode_string(80)
            ),

        "col_string_3":
            maybe_null(
                lambda:
                random_unicode_string(40)
            ),

        "col_string_4":
            maybe_null(
                random_lob_text
            ),

        "col_string_5":
            maybe_null(
                random_lob_text
            ),


        # DATETIME
        "col_datetime_0":
            maybe_null(
                random_datetime
            ),

        "col_datetime_1":
            maybe_null(
                random_datetime
            ),

        "col_datetime_2":
            maybe_null(
                random_timestamp_tz_text
            ),

        "col_datetime_3":
            maybe_null(
                random_timestamp_tz_text
            ),


        # INTERVAL
        "col_interval_0":
            maybe_null(
                random_interval_ym
            ),

        "col_interval_1":
            maybe_null(
                random_interval_ds
            ),


        # BINARY
        "col_binary_0":
            maybe_null(
                lambda:
                random_raw(32)
            ),

        "col_binary_1":
            maybe_null(
                lambda:
                random_blob(256)
            ),


        # BOOLEAN
        "col_boolean_0":
            maybe_null(
                random_boolean_text
            ),


        # JSON
        "col_json_0":
            maybe_null(
                random_json_text
            ),


        # XML
        "col_xml_0":
            maybe_null(
                random_xml_text
            ),


        # VECTOR
        "col_vector_0":
            maybe_null(
                random_vector_text
            ),
    }


# ============================================================
# 主程序
# ============================================================

def main():

    if not DB_CONFIG.password:

        raise RuntimeError(
            "RL/config.py 中的 DB_CONFIG.password 为空，请先填写数据库连接配置。"
        )

    print(
        "[Initializer] "
        f"Connecting to Oracle "
        f"{DB_CONFIG.host}:{DB_CONFIG.port}/{DB_CONFIG.database}"
    )

    conn = oracledb.connect(

        user=DB_CONFIG.user,

        password=DB_CONFIG.password,

        host=DB_CONFIG.host,

        port=DB_CONFIG.port,

        service_name=DB_CONFIG.database,
        tcp_connect_timeout=DB_CONFIG.connect_timeout,
    )

    try:

        with conn.cursor() as cursor:

            # ------------------------------------
            # 建表
            # ------------------------------------

            print(
                "[Initializer] "
                f"Creating table: {TABLE_NAME}"
            )

            cursor.execute(
                CREATE_TABLE_SQL
            )

            print(
                "[Initializer] "
                "Table created."
            )

            # 明确指定特殊类型的 bind 类型，避免 NULL 与 bytes 混用时被误判为字符串
            cursor.setinputsizes(
                col_binary_0=oracledb.DB_TYPE_RAW,
                col_binary_1=oracledb.DB_TYPE_BLOB,
                col_string_4=oracledb.DB_TYPE_CLOB,
                col_string_5=oracledb.DB_TYPE_NCLOB,
            )


            # ------------------------------------
            # 插入 500 行
            # ------------------------------------

            print(
                "[Initializer] "
                f"Inserting {INITIAL_ROWS} rows, "
                f"NULL probability = "
                f"{NULL_PROBABILITY:.0%}"
            )

            for i in range(
                1,
                INITIAL_ROWS + 1
            ):

                cursor.execute(
                    INSERT_SQL,
                    make_row()
                )

                if (
                    i % COMMIT_EVERY == 0
                ):

                    conn.commit()

                    print(
                        "[Initializer] "
                        f"Inserted "
                        f"{i}/{INITIAL_ROWS}"
                    )

            conn.commit()


            # ------------------------------------
            # 验证最终行数
            # ------------------------------------

            cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM {TABLE_SQL}
                """
            )

            row_count = (
                cursor.fetchone()[0]
            )


            # ------------------------------------
            # 验证最终列数
            # ------------------------------------

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM user_tab_columns
                WHERE table_name = :table_name
                """,

                table_name=
                    TABLE_NAME.upper()
            )

            column_count = (
                cursor.fetchone()[0]
            )


            print(
                "[Initializer] "
                "Initialization completed."
            )

            print(
                "[Initializer] "
                f"Rows = {row_count}"
            )

            print(
                "[Initializer] "
                f"Columns = {column_count}"
            )


    except Exception:

        conn.rollback()

        raise


    finally:

        conn.close()

        print(
            "[Initializer] "
            "Oracle connection closed."
        )


if __name__ == "__main__":

    main()
