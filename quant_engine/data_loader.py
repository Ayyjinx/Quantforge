import os
import sqlite3

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------
# Persistent storage
# ---------------------------------------------------------

DATABASE_DIRECTORY = r"D:\QuantForgeData"

DATABASE_PATH = os.path.join(
    DATABASE_DIRECTORY,
    "market_data.db"
)


# ---------------------------------------------------------
# Database initialization
# ---------------------------------------------------------

def initialize_database():

    os.makedirs(
        DATABASE_DIRECTORY,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS market_data (

            instrument TEXT NOT NULL,

            timeframe TEXT NOT NULL,

            timestamp TEXT NOT NULL,

            open REAL NOT NULL,

            high REAL NOT NULL,

            low REAL NOT NULL,

            close REAL NOT NULL,

            volume REAL DEFAULT 0,

            PRIMARY KEY (
                instrument,
                timeframe,
                timestamp
            )
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_market_data_lookup

        ON market_data (
            instrument,
            timeframe,
            timestamp
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS
        data_metadata (

            instrument TEXT NOT NULL,

            timeframe TEXT NOT NULL,

            oldest_timestamp TEXT,

            newest_timestamp TEXT,

            last_updated TEXT NOT NULL,

            PRIMARY KEY (
                instrument,
                timeframe
            )
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS
        research_data_sessions (

            research_run_id TEXT PRIMARY KEY,

            instrument TEXT NOT NULL,

            timeframe TEXT NOT NULL,

            prepared_at TEXT NOT NULL,

            oldest_timestamp TEXT,

            newest_timestamp TEXT
        )
        """
    )

    connection.commit()

    connection.close()


# ---------------------------------------------------------
# Instrument mapping
# ---------------------------------------------------------

def get_yahoo_ticker(
    instrument
):

    instrument = str(
        instrument
    ).upper().strip()

    ticker_map = {

        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",
        "USDCHF": "CHF=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "CAD=X",
        "NZDUSD": "NZDUSD=X"

    }

    if instrument not in ticker_map:

        raise ValueError(
            "Unsupported Forex instrument: " +
            instrument
        )

    return ticker_map[
        instrument
    ]


# ---------------------------------------------------------
# Timeframe mapping
# ---------------------------------------------------------

def normalize_timeframe(
    timeframe
):

    timeframe = str(
        timeframe
    ).lower().strip()

    supported = {

        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d"

    }

    if timeframe not in supported:

        raise ValueError(
            "Unsupported timeframe: " +
            timeframe +
            ". Supported timeframes: " +
            ", ".join(
                supported.keys()
            )
        )

    return supported[
        timeframe
    ]


# ---------------------------------------------------------
# Historical period selection
# ---------------------------------------------------------

def get_download_period(
    timeframe
):

    timeframe = normalize_timeframe(
        timeframe
    )

    period_map = {

        "15m": "60d",
        "30m": "60d",
        "1h": "2y",
        "4h": "2y",
        "1d": "10y"

    }

    return period_map[
        timeframe
    ]


# ---------------------------------------------------------
# Resampling
# ---------------------------------------------------------

def resample_data(
    data,
    timeframe
):

    if timeframe != "4h":

        return data

    resampled = data.resample(
        "4h"
    ).agg({

        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"

    })

    return resampled.dropna()


# ---------------------------------------------------------
# Normalize downloaded data
# ---------------------------------------------------------

def normalize_downloaded_data(
    data,
    timeframe
):

    if data is None or data.empty:

        raise ValueError(
            "Downloaded market data is empty."
        )

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = [
            column[0]
            for column in data.columns
        ]

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    for column in required_columns:

        if column not in data.columns:

            raise ValueError(
                "Downloaded data is missing " +
                column
            )

    if "Volume" not in data.columns:

        data["Volume"] = 0

    data = data[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ]

    data = data.dropna()

    # Yahoo Finance timestamps can be timezone-aware while
    # previously cached/database timestamps may use a different
    # timezone representation. Normalize the index to UTC before
    # any resampling or database operations.
    try:
        data.index = pd.to_datetime(
            data.index,
            utc=True
        )
    except Exception as error:
        raise ValueError(
            "Unable to normalize market-data timestamps to UTC: "
            + str(error)
        )

    data = resample_data(
        data,
        timeframe
    )

    if data.empty:

        raise ValueError(
            "No usable market data remains " +
            "after preprocessing."
        )

    return data


# ---------------------------------------------------------
# Store market data
# ---------------------------------------------------------

def store_market_data(
    data,
    instrument,
    timeframe
):

    initialize_database()

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    rows = []

    for timestamp, row in data.iterrows():

        timestamp_string = (
            pd.Timestamp(
                timestamp
            )
            .tz_convert("UTC")
            .isoformat()
        )

        rows.append(
            (
                instrument,
                timeframe,
                timestamp_string,
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                float(row["Volume"])
            )
        )

    connection.executemany(
        """
        INSERT OR REPLACE INTO market_data (
            instrument,
            timeframe,
            timestamp,
            open,
            high,
            low,
            close,
            volume
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows
    )

    oldest_timestamp = (
        pd.Timestamp(
            data.index.min()
        )
        .tz_convert("UTC")
        .isoformat()
    )

    newest_timestamp = (
        pd.Timestamp(
            data.index.max()
        )
        .tz_convert("UTC")
        .isoformat()
    )

    last_updated = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    connection.execute(
        """
        INSERT OR REPLACE INTO data_metadata (
            instrument,
            timeframe,
            oldest_timestamp,
            newest_timestamp,
            last_updated
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            instrument,
            timeframe,
            oldest_timestamp,
            newest_timestamp,
            last_updated
        )
    )

    connection.commit()

    connection.close()


# ---------------------------------------------------------
# Read market data
# ---------------------------------------------------------

def read_market_data(
    instrument,
    timeframe
):

    initialize_database()

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    query = """
        SELECT
            timestamp,
            open AS Open,
            high AS High,
            low AS Low,
            close AS Close,
            volume AS Volume
        FROM market_data
        WHERE instrument = ?
        AND timeframe = ?
        ORDER BY timestamp ASC
    """

    data = pd.read_sql_query(
        query,
        connection,
        params=(
            instrument,
            timeframe
        )
    )

    connection.close()

    if data.empty:

        return pd.DataFrame()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True
    )

    data = data.set_index(
        "timestamp"
    )

    return data


# ---------------------------------------------------------
# Read metadata
# ---------------------------------------------------------

def read_metadata(
    instrument,
    timeframe
):

    initialize_database()

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    query = """
        SELECT
            oldest_timestamp,
            newest_timestamp,
            last_updated
        FROM data_metadata
        WHERE instrument = ?
        AND timeframe = ?
    """

    result = connection.execute(
        query,
        (
            instrument,
            timeframe
        )
    ).fetchone()

    connection.close()

    if result is None:

        return None

    return {
        "oldest_timestamp": result[0],
        "newest_timestamp": result[1],
        "last_updated": result[2]
    }


# ---------------------------------------------------------
# Read research data session
# ---------------------------------------------------------

def read_research_session(
    research_run_id
):

    initialize_database()

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    query = """
        SELECT
            research_run_id,
            instrument,
            timeframe,
            prepared_at,
            oldest_timestamp,
            newest_timestamp
        FROM research_data_sessions
        WHERE research_run_id = ?
    """

    result = connection.execute(
        query,
        (
            research_run_id,
        )
    ).fetchone()

    connection.close()

    if result is None:

        return None

    return {

        "research_run_id": result[0],

        "instrument": result[1],

        "timeframe": result[2],

        "prepared_at": result[3],

        "oldest_timestamp": result[4],

        "newest_timestamp": result[5]

    }


# ---------------------------------------------------------
# Create research data session
# ---------------------------------------------------------

def create_research_session(
    research_run_id,
    instrument,
    timeframe,
    data
):

    initialize_database()

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    prepared_at = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    oldest_timestamp = (
        pd.Timestamp(
            data.index.min()
        )
        .tz_convert("UTC")
        .isoformat()
    )

    newest_timestamp = (
        pd.Timestamp(
            data.index.max()
        )
        .tz_convert("UTC")
        .isoformat()
    )

    connection.execute(
        """
        INSERT OR REPLACE INTO
        research_data_sessions (
            research_run_id,
            instrument,
            timeframe,
            prepared_at,
            oldest_timestamp,
            newest_timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            research_run_id,
            instrument,
            timeframe,
            prepared_at,
            oldest_timestamp,
            newest_timestamp
        )
    )

    connection.commit()

    connection.close()


# ---------------------------------------------------------
# Check whether cached data is fresh
# ---------------------------------------------------------

def is_data_fresh(
    instrument,
    timeframe
):

    metadata = read_metadata(
        instrument,
        timeframe
    )

    if metadata is None:

        return False

    newest_timestamp = pd.Timestamp(
        metadata[
            "newest_timestamp"
        ],
        tz="UTC"
    )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    freshness_minutes = {

        "15m": 30,
        "30m": 60,
        "1h": 120,
        "4h": 360,
        "1d": 2880

    }

    threshold = pd.Timedelta(
        minutes=freshness_minutes[
            timeframe
        ]
    )

    return (
        now - newest_timestamp
    ) <= threshold


# ---------------------------------------------------------
# Prepare market data for research run
# ---------------------------------------------------------

def prepare_research_data(
    research_run_id,
    instrument,
    timeframe
):

    research_run_id = str(
        research_run_id
    ).strip()

    if not research_run_id:

        raise ValueError(
            "research_run_id is required."
        )

    instrument = str(
        instrument
    ).upper().strip()

    timeframe = normalize_timeframe(
        timeframe
    )

    initialize_database()

    existing_session = read_research_session(
        research_run_id
    )

    if existing_session is not None:

        if (
            existing_session["instrument"]
            != instrument
            or
            existing_session["timeframe"]
            != timeframe
        ):

            raise ValueError(
                "Research run already exists with "
                "a different instrument or timeframe."
            )

        print(
            "Research data session already prepared."
        )

        print(
            "Research run: " +
            research_run_id
        )

        print(
            "Reusing existing market dataset."
        )

        data = read_market_data(
            instrument,
            timeframe
        )

        if data.empty:

            raise ValueError(
                "Research session exists but "
                "market data is unavailable."
            )

        return data

    print(
        "Preparing market data for new research run."
    )

    print(
        "Research run: " +
        research_run_id
    )

    data = read_market_data(
        instrument,
        timeframe
    )

    if data.empty:

        print(
            "No cached market data found."
        )

        data = download_market_data(
            instrument,
            timeframe
        )

    elif is_data_fresh(
        instrument,
        timeframe
    ):

        print(
            "Cached market data is fresh."
        )

    else:

        print(
            "Cached market data is stale."
        )

        print(
            "Refreshing market data once "
            "for this research run."
        )

        data = download_market_data(
            instrument,
            timeframe
        )

    create_research_session(
        research_run_id,
        instrument,
        timeframe,
        data
    )

    print(
        "Research data session prepared."
    )

    print(
        "Rows available: " +
        str(len(data))
    )

    return data


# ---------------------------------------------------------
# Download market data
# ---------------------------------------------------------

def download_market_data(
    instrument,
    timeframe,
    period=None
):

    instrument = str(
        instrument
    ).upper().strip()

    timeframe = normalize_timeframe(
        timeframe
    )

    ticker = get_yahoo_ticker(
        instrument
    )

    download_interval = (
        "1h"
        if timeframe == "4h"
        else timeframe
    )

    if period is None:

        period = get_download_period(
            timeframe
        )

    print(
        "Downloading " +
        instrument +
        " " +
        timeframe +
        " data..."
    )

    print(
        "Yahoo Finance period: " +
        str(period)
    )

    print(
        "Yahoo Finance interval: " +
        download_interval
    )

    data = yf.download(
        ticker,
        period=period,
        interval=download_interval,
        auto_adjust=False,
        progress=False
    )

    data = normalize_downloaded_data(
        data,
        timeframe
    )

    store_market_data(
        data,
        instrument,
        timeframe
    )

    print(
        "Stored market data in: " +
        DATABASE_PATH
    )

    print(
        "Rows processed: " +
        str(len(data))
    )

    return data


# ---------------------------------------------------------
# Main data loader
# ---------------------------------------------------------

def load_market_data(
    instrument,
    timeframe,
    research_run_id=None
):

    instrument = str(
        instrument
    ).upper().strip()

    timeframe = normalize_timeframe(
        timeframe
    )

    if research_run_id is None:

        raise ValueError(
            "research_run_id is required "
            "for market data loading."
        )

    return prepare_research_data(
        research_run_id=research_run_id,
        instrument=instrument,
        timeframe=timeframe
    )