import json
import os
import random
import string
import sys
import uuid
from datetime import datetime, timedelta
from html import escape as xml_escape
from pathlib import Path

import pyodbc


project_root = str(Path(__file__).resolve().parents[2] / "src")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from RL.config import DB_CONFIG, get_target_table


TABLE_NAME, TABLE_SCHEMA, QUALIFIED_TABLE = get_target_table("sqlserver", DB_CONFIG)
JSON_CONSTRAINT_NAME = "[" + ("CK_" + TABLE_NAME + "_json").replace("]", "]]") + "]"

INITIAL_ROWS = 500
NULL_PROBABILITY = 0.15
PROGRESS_EVERY = 100


def maybe_null(generator):
    if random.random() < NULL_PROBABILITY:
        return None
    return generator()


def choose_special_or_random(special_values, random_generator, p=0.25):
    if random.random() < p:
        return random.choice(special_values)
    return random_generator()


def random_tinyint():
    return choose_special_or_random(
        [0, 1, 127, 254, 255],
        lambda: random.randint(0, 255),
    )


def random_smallint():
    return choose_special_or_random(
        [-32768, -1, 0, 1, 32767],
        lambda: random.randint(-32768, 32767),
    )


def random_int():
    return choose_special_or_random(
        [-2147483648, -10000, -1, 0, 1, 10000, 2147483647],
        lambda: random.randint(-1000000, 1000000),
    )


def random_bigint():
    return choose_special_or_random(
        [
            -9223372036854775808,
            -(10**18),
            -1,
            0,
            1,
            10**18,
            9223372036854775807,
        ],
        lambda: random.randint(-(10**15), 10**15),
    )


def random_bit():
    return random.choice([0, 1])


def random_decimal_18_6_text():
    special = [
        "-999999999999.999999",
        "-1.000000",
        "0.000000",
        "1.000000",
        "999999999999.999999",
    ]

    def normal():
        whole = random.randint(-10000000, 10000000)
        frac = random.randint(0, 999999)
        sign = "-" if whole < 0 else ""
        return f"{whole}.{frac:06d}" if whole != 0 else f"{sign}0.{frac:06d}"

    return choose_special_or_random(special, normal)


def random_numeric_38_10_text():
    special = [
        "-9999999999999999999999999999.9999999999",
        "-1.0000000000",
        "0.0000000000",
        "1.0000000000",
        "9999999999999999999999999999.9999999999",
    ]

    def normal():
        whole = random.randint(-(10**15), 10**15)
        frac = random.randint(0, 9999999999)
        return f"{whole}.{frac:010d}"

    return choose_special_or_random(special, normal)


def random_smallmoney_text():
    return choose_special_or_random(
        ["-214748.3648", "-1.0000", "0.0000", "1.0000", "214748.3647"],
        lambda: f"{random.uniform(-100000, 100000):.4f}",
    )


def random_money_text():
    return choose_special_or_random(
        [
            "-922337203685477.5808",
            "-1.0000",
            "0.0000",
            "1.0000",
            "922337203685477.5807",
        ],
        lambda: f"{random.uniform(-100000000, 100000000):.4f}",
    )


def random_real():
    return choose_special_or_random(
        [-1.0e30, -10000.0, -1.0, -0.0, 0.0, 1.0, 10000.0, 1.0e30],
        lambda: random.uniform(-10000.0, 10000.0),
    )


def random_float():
    return choose_special_or_random(
        [-1.0e200, -10000.0, -1.0, -0.0, 0.0, 1.0, 10000.0, 1.0e200],
        lambda: random.uniform(-10000.0, 10000.0),
    )


ASCII_CHARS = (
    string.ascii_letters
    + string.digits
    + " _-@'.,:/%[]+"
)


def random_ascii_string(max_len=80):
    special_values = [
        "",
        " ",
        "a",
        "A",
        "0",
        "NULL",
        "true",
        "false",
        "100%",
        "_underscore_",
        "[brackets]",
        "single'quote",
        " leading",
        "trailing ",
    ]

    if random.random() < 0.30:
        return random.choice(special_values)

    length = random.randint(1, max_len)
    return "".join(random.choice(ASCII_CHARS) for _ in range(length))


def random_unicode_string(max_len=80):
    special_values = [
        "",
        "测试",
        "数据库",
        "SQL Server",
        "中文English混合",
        "空 格",
        "全角ＡＢＣ１２３",
        "符号_-=+@#%",
        "换行\n测试",
        "制表\t测试",
        "éàöñ",
        "Δοκιμή",
        "данные",
        "データ",
        "😀",
    ]

    if random.random() < 0.40:
        return random.choice(special_values)

    return random_ascii_string(max_len)


def random_long_ascii_text():
    return " | ".join(
        random_ascii_string(120)
        for _ in range(random.randint(4, 12))
    )


def random_long_unicode_text():
    return " | ".join(
        random_unicode_string(120)
        for _ in range(random.randint(4, 12))
    )


def random_recent_datetime():
    start = datetime.now() - timedelta(days=365 * 10)
    end = datetime.now()
    return start + (end - start) * random.random()


def random_date_text():
    special = [
        "0001-01-01",
        "1900-01-01",
        "1970-01-01",
        "2000-02-29",
        "9999-12-31",
    ]

    return choose_special_or_random(
        special,
        lambda: random_recent_datetime().strftime("%Y-%m-%d"),
    )


def random_time_text():
    special = [
        "00:00:00.000000",
        "00:00:00.000001",
        "12:00:00.000000",
        "23:59:59.999999",
    ]

    def normal():
        return (
            f"{random.randint(0, 23):02d}:"
            f"{random.randint(0, 59):02d}:"
            f"{random.randint(0, 59):02d}."
            f"{random.randint(0, 999999):06d}"
        )

    return choose_special_or_random(special, normal)


def random_datetime_text():
    special = [
        "1753-01-01T00:00:00.000",
        "1900-01-01T00:00:00.000",
        "2000-02-29T12:34:56.000",
        "9999-12-31T23:59:59.997",
    ]

    return choose_special_or_random(
        special,
        lambda: random_recent_datetime().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
    )


def random_smalldatetime_text():
    special = [
        "1900-01-01T00:00:00",
        "2000-02-29T12:34:00",
        "2079-06-06T23:59:00",
    ]

    return choose_special_or_random(
        special,
        lambda: random_recent_datetime().strftime("%Y-%m-%dT%H:%M:00"),
    )


def random_datetime2_text():
    special = [
        "0001-01-01T00:00:00.000000",
        "1900-01-01T00:00:00.000000",
        "2000-02-29T12:34:56.123456",
        "9999-12-31T23:59:59.999999",
    ]

    return choose_special_or_random(
        special,
        lambda: random_recent_datetime().strftime("%Y-%m-%dT%H:%M:%S.%f"),
    )


def random_datetimeoffset_text():
    if random.random() < 0.25:
        return random.choice(
            [
                "2000-01-01T00:00:00.000000+00:00",
                "2000-01-01T08:00:00.000000+08:00",
                "2000-01-01T01:00:00.000000-05:00",
                "2024-02-29T23:59:59.999999+05:30",
            ]
        )

    dt = random_recent_datetime()
    offset_minutes = random.choice(
        [-480, -300, 0, 60, 330, 480, 540]
    )
    sign = "+" if offset_minutes >= 0 else "-"
    hours, minutes = divmod(abs(offset_minutes), 60)

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        + f"{sign}{hours:02d}:{minutes:02d}"
    )


def random_binary(max_len=64):
    special = [
        b"",
        b"\x00",
        b"\xff",
        b"\x00\xff\x00\xff",
    ]

    if random.random() < 0.25:
        return random.choice(special)

    return os.urandom(random.randint(1, max_len))


def random_uuid_text():
    if random.random() < 0.20:
        return random.choice(
            [
                "00000000-0000-0000-0000-000000000000",
                "ffffffff-ffff-ffff-ffff-ffffffffffff",
            ]
        )
    return str(uuid.uuid4())


def random_json_text():
    obj = {
        "id": random.randint(1, 1000),
        "tag": "".join(random.choices(string.ascii_lowercase, k=5)),
        "score": round(random.uniform(-10, 10), 4),
        "active": random.choice([True, False]),
        "note": random_unicode_string(30),
        "nullable": random.choice([None, 0, "", "value"]),
        "items": [
            random.randint(-5, 5)
            for _ in range(random.randint(0, 5))
        ],
        "nested": {
            "x": random.randint(-100, 100),
            "y": random_unicode_string(15),
        },
    }
    return json.dumps(obj, ensure_ascii=False)


def random_xml_text():
    note = xml_escape(random_unicode_string(30))
    tag = "".join(random.choices(string.ascii_letters, k=8))
    score = round(random.uniform(-10, 10), 4)

    return (
        "<row>"
        f"<id>{random.randint(1, 10000)}</id>"
        f"<tag>{tag}</tag>"
        f"<score>{score}</score>"
        f"<note>{note}</note>"
        "</row>"
    )


def random_hierarchyid_text():
    if random.random() < 0.20:
        return "/"

    depth = random.randint(1, 4)
    labels = [
        str(random.randint(1, 20))
        for _ in range(depth)
    ]
    return "/" + "/".join(labels) + "/"


def random_geometry_wkt():
    kind = random.choice(["POINT", "LINESTRING", "POLYGON"])

    if kind == "POINT":
        x = round(random.uniform(-1000, 1000), 6)
        y = round(random.uniform(-1000, 1000), 6)
        return f"POINT ({x} {y})"

    if kind == "LINESTRING":
        x1 = round(random.uniform(-1000, 1000), 6)
        y1 = round(random.uniform(-1000, 1000), 6)
        x2 = round(random.uniform(-1000, 1000), 6)
        y2 = round(random.uniform(-1000, 1000), 6)
        return f"LINESTRING ({x1} {y1}, {x2} {y2})"

    x = round(random.uniform(-100, 100), 6)
    y = round(random.uniform(-100, 100), 6)
    w = round(random.uniform(0.1, 20), 6)
    h = round(random.uniform(0.1, 20), 6)
    return (
        "POLYGON (("
        f"{x} {y}, "
        f"{x + w} {y}, "
        f"{x + w} {y + h}, "
        f"{x} {y + h}, "
        f"{x} {y}"
        "))"
    )


def random_geography_wkt():

    longitude = round(random.uniform(-179.0, 179.0), 6)
    latitude = round(random.uniform(-89.0, 89.0), 6)
    return f"POINT ({longitude} {latitude})"


def random_sql_variant():


    generators = [
        ("CAST(CAST(? AS INT) AS SQL_VARIANT)", random_int),
        ("CAST(CAST(? AS BIGINT) AS SQL_VARIANT)", random_bigint),
        (
            "CAST(CAST(? AS DECIMAL(18,6)) AS SQL_VARIANT)",
            random_decimal_18_6_text,
        ),
        (
            "CAST(CAST(? AS NVARCHAR(100)) AS SQL_VARIANT)",
            lambda: random_unicode_string(80),
        ),
        (
            "CAST(CAST(? AS VARBINARY(32)) AS SQL_VARIANT)",
            lambda: random_binary(32),
        ),
        ("CAST(CAST(? AS DATE) AS SQL_VARIANT)", random_date_text),
        (
            "CAST(CAST(? AS DATETIME2(6)) AS SQL_VARIANT)",
            random_datetime2_text,
        ),
        (
            "CAST(CAST(? AS UNIQUEIDENTIFIER) AS SQL_VARIANT)",
            random_uuid_text,
        ),
    ]

    expression, generator = random.choice(generators)

    if random.random() < NULL_PROBABILITY:
        return expression, None

    return expression, generator()


CREATE_TABLE_SQL = f"""
CREATE TABLE {QUALIFIED_TABLE} (

    id BIGINT
        IDENTITY(1,1)
        PRIMARY KEY,

    -- ================= Integer / Boolean =================

    col_integer_0 TINYINT NULL,
    col_integer_1 SMALLINT NULL,
    col_integer_2 INT NULL,
    col_integer_3 BIGINT NULL,
    col_boolean_0 BIT NULL,

    -- ================= Exact numeric =================

    col_decimal_0 DECIMAL(18,6) NULL,
    col_decimal_1 NUMERIC(38,10) NULL,
    col_money_0 SMALLMONEY NULL,
    col_money_1 MONEY NULL,

    -- ================= Approximate numeric =================

    col_float_0 REAL NULL,
    col_float_1 FLOAT(53) NULL,

    -- ================= Character / Unicode =================

    col_string_0 CHAR(50) NULL,
    col_string_1 VARCHAR(255) NULL,
    col_string_2 VARCHAR(MAX) NULL,

    col_string_3 NCHAR(50) NULL,
    col_string_4 NVARCHAR(255) NULL,
    col_string_5 NVARCHAR(MAX) NULL,

    -- Legacy types: still supported by SQL Server 2022,
    -- but deprecated by Microsoft.
    col_string_6 TEXT NULL,
    col_string_7 NTEXT NULL,

    -- ================= Date / Time =================

    col_datetime_0 DATE NULL,
    col_datetime_1 TIME(6) NULL,
    col_datetime_2 DATETIME NULL,
    col_datetime_3 SMALLDATETIME NULL,
    col_datetime_4 DATETIME2(6) NULL,
    col_datetime_5 DATETIMEOFFSET(6) NULL,

    -- ================= Binary =================

    col_binary_0 BINARY(64) NULL,
    col_binary_1 VARBINARY(256) NULL,
    col_binary_2 VARBINARY(MAX) NULL,

    -- Legacy binary LOB.
    col_binary_3 IMAGE NULL,

    -- Automatically generated by SQL Server.
    col_rowversion_0 ROWVERSION,

    -- ================= SQL Server-specific =================

    col_uuid_0 UNIQUEIDENTIFIER NULL,

    col_variant_0 SQL_VARIANT NULL,

    col_xml_0 XML NULL,

    -- SQL Server 2022 has JSON functions, but no native JSON column type.
    col_json_0 NVARCHAR(MAX) NULL,

    col_hierarchy_0 HIERARCHYID NULL,

    col_geometry_0 GEOMETRY NULL,

    col_geography_0 GEOGRAPHY NULL,

    CONSTRAINT {JSON_CONSTRAINT_NAME}
        CHECK (
            col_json_0 IS NULL
            OR ISJSON(col_json_0) = 1
        )
)
"""


INSERT_COLUMNS = [
    "col_integer_0",
    "col_integer_1",
    "col_integer_2",
    "col_integer_3",
    "col_boolean_0",

    "col_decimal_0",
    "col_decimal_1",
    "col_money_0",
    "col_money_1",

    "col_float_0",
    "col_float_1",

    "col_string_0",
    "col_string_1",
    "col_string_2",
    "col_string_3",
    "col_string_4",
    "col_string_5",
    "col_string_6",
    "col_string_7",

    "col_datetime_0",
    "col_datetime_1",
    "col_datetime_2",
    "col_datetime_3",
    "col_datetime_4",
    "col_datetime_5",

    "col_binary_0",
    "col_binary_1",
    "col_binary_2",
    "col_binary_3",


    "col_uuid_0",
    "col_variant_0",
    "col_xml_0",
    "col_json_0",
    "col_hierarchy_0",
    "col_geometry_0",
    "col_geography_0",
]


STATIC_VALUE_EXPRESSIONS = {
    "col_decimal_0": "CAST(? AS DECIMAL(18,6))",
    "col_decimal_1": "CAST(? AS NUMERIC(38,10))",
    "col_money_0": "CAST(? AS SMALLMONEY)",
    "col_money_1": "CAST(? AS MONEY)",

    "col_datetime_0": "CAST(? AS DATE)",
    "col_datetime_1": "CAST(? AS TIME(6))",
    "col_datetime_2": "CAST(? AS DATETIME)",
    "col_datetime_3": "CAST(? AS SMALLDATETIME)",
    "col_datetime_4": "CAST(? AS DATETIME2(6))",
    "col_datetime_5": "CAST(? AS DATETIMEOFFSET(6))",

    "col_uuid_0": "CAST(? AS UNIQUEIDENTIFIER)",
    "col_xml_0": "CAST(? AS XML)",

    "col_hierarchy_0": "HIERARCHYID::Parse(CAST(? AS NVARCHAR(4000)))",

    "col_geometry_0": (
        "GEOMETRY::STGeomFromText("
        "CAST(? AS NVARCHAR(MAX)), 0)"
    ),

    "col_geography_0": (
        "GEOGRAPHY::STGeomFromText("
        "CAST(? AS NVARCHAR(MAX)), 4326)"
    ),
}


def make_row():
    return {
        "col_integer_0": maybe_null(random_tinyint),
        "col_integer_1": maybe_null(random_smallint),
        "col_integer_2": maybe_null(random_int),
        "col_integer_3": maybe_null(random_bigint),
        "col_boolean_0": maybe_null(random_bit),

        "col_decimal_0": maybe_null(random_decimal_18_6_text),
        "col_decimal_1": maybe_null(random_numeric_38_10_text),
        "col_money_0": maybe_null(random_smallmoney_text),
        "col_money_1": maybe_null(random_money_text),

        "col_float_0": maybe_null(random_real),
        "col_float_1": maybe_null(random_float),

        "col_string_0": maybe_null(lambda: random_ascii_string(40)),
        "col_string_1": maybe_null(lambda: random_ascii_string(120)),
        "col_string_2": maybe_null(random_long_ascii_text),

        "col_string_3": maybe_null(lambda: random_unicode_string(40)),
        "col_string_4": maybe_null(lambda: random_unicode_string(120)),
        "col_string_5": maybe_null(random_long_unicode_text),

        "col_string_6": maybe_null(random_long_ascii_text),
        "col_string_7": maybe_null(random_long_unicode_text),

        "col_datetime_0": maybe_null(random_date_text),
        "col_datetime_1": maybe_null(random_time_text),
        "col_datetime_2": maybe_null(random_datetime_text),
        "col_datetime_3": maybe_null(random_smalldatetime_text),
        "col_datetime_4": maybe_null(random_datetime2_text),
        "col_datetime_5": maybe_null(random_datetimeoffset_text),

        "col_binary_0": maybe_null(lambda: random_binary(64)),
        "col_binary_1": maybe_null(lambda: random_binary(256)),
        "col_binary_2": maybe_null(lambda: random_binary(512)),
        "col_binary_3": maybe_null(lambda: random_binary(512)),

        "col_uuid_0": maybe_null(random_uuid_text),


        "col_variant_0": random_sql_variant(),

        "col_xml_0": maybe_null(random_xml_text),
        "col_json_0": maybe_null(random_json_text),

        "col_hierarchy_0": maybe_null(random_hierarchyid_text),
        "col_geometry_0": maybe_null(random_geometry_wkt),
        "col_geography_0": maybe_null(random_geography_wkt),
    }


def insert_row(cursor, row):
    value_expressions = []
    parameters = []

    for column_name in INSERT_COLUMNS:

        if column_name == "col_variant_0":
            expression, value = row[column_name]
            value_expressions.append(expression)
            parameters.append(value)
            continue

        value = row[column_name]


        if value is None and column_name == "col_hierarchy_0":
            value_expressions.append("CAST(NULL AS HIERARCHYID)")
            continue

        if value is None and column_name == "col_geometry_0":
            value_expressions.append("CAST(NULL AS GEOMETRY)")
            continue

        if value is None and column_name == "col_geography_0":
            value_expressions.append("CAST(NULL AS GEOGRAPHY)")
            continue

        expression = STATIC_VALUE_EXPRESSIONS.get(
            column_name,
            "?"
        )

        value_expressions.append(expression)
        parameters.append(value)

    column_sql = ",\n    ".join(INSERT_COLUMNS)
    values_sql = ",\n    ".join(value_expressions)

    sql = f"""
    INSERT INTO {QUALIFIED_TABLE} (
        {column_sql}
    )
    VALUES (
        {values_sql}
    )
    """

    cursor.execute(sql, parameters)


def main():
    if not DB_CONFIG.password:
        raise RuntimeError(
            "Database password is empty. "
            "Fill in Robusta/configuration.json."
        )

    connection_values = {
        "DRIVER": getattr(DB_CONFIG, "odbc_driver", "ODBC Driver 18 for SQL Server"),
        "SERVER": f"{DB_CONFIG.host},{DB_CONFIG.port}",
        "DATABASE": DB_CONFIG.database,
        "UID": DB_CONFIG.user,
        "PWD": DB_CONFIG.password,
    }
    connection_string = "".join(
        key + "={" + str(value).replace("}", "}}") + "};"
        for key, value in connection_values.items()
    ) + (
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        f"Connection Timeout={DB_CONFIG.connect_timeout};"
    )

    print(
        "[Initializer] Connecting to SQL Server "
        f"{DB_CONFIG.host}:{DB_CONFIG.port}/{DB_CONFIG.database}"
    )

    conn = pyodbc.connect(
        connection_string,
        autocommit=False,
    )

    try:
        cursor = conn.cursor()


        cursor.execute(
            """
            SELECT OBJECT_ID(?, 'U')
            """,
            QUALIFIED_TABLE,
        )

        existing_object_id = cursor.fetchone()[0]

        if existing_object_id is not None:
            cursor.execute(f"DROP TABLE {QUALIFIED_TABLE}")


        print(
            "[Initializer] Creating table: "
            f"{TABLE_SCHEMA}.{TABLE_NAME}"
        )

        cursor.execute(CREATE_TABLE_SQL)

        print("[Initializer] Table created.")


        print(
            "[Initializer] "
            f"Inserting {INITIAL_ROWS} rows, "
            f"NULL probability = {NULL_PROBABILITY:.0%}"
        )

        for i in range(1, INITIAL_ROWS + 1):
            insert_row(cursor, make_row())

            if i % PROGRESS_EVERY == 0:
                print(
                    "[Initializer] "
                    f"Generated and inserted "
                    f"{i}/{INITIAL_ROWS}"
                )


        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {QUALIFIED_TABLE}
            """
        )
        row_count = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM sys.columns
            WHERE object_id = OBJECT_ID(?, 'U')
            """,
            QUALIFIED_TABLE,
        )
        column_count = cursor.fetchone()[0]


        conn.commit()

        print("[Initializer] Initialization completed.")
        print(f"[Initializer] Rows = {row_count}")
        print(f"[Initializer] Columns = {column_count}")

        cursor.close()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()
        print("[Initializer] SQL Server connection closed.")


if __name__ == "__main__":
    main()
