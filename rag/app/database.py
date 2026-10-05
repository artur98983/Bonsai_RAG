import sqlite3

from .config import DATABASE_PATH


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'automatic'
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            file TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            type TEXT NOT NULL,
            language TEXT NOT NULL,
            parent_symbol_id INTEGER,
            start_line INTEGER,
            end_line INTEGER,
            code TEXT,
            FOREIGN KEY(project_id)
                REFERENCES projects(id)
                ON DELETE CASCADE,
            FOREIGN KEY(parent_symbol_id)
                REFERENCES symbols(id)
                ON DELETE SET NULL
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbols_project
        ON symbols(project_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbols_name
        ON symbols(project_id, name)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbols_qualified_name
        ON symbols(project_id, qualified_name)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbols_parent
        ON symbols(parent_symbol_id)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS code_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            source_symbol_id INTEGER,
            target_symbol_id INTEGER,
            target_name TEXT NOT NULL,
            reference_type TEXT NOT NULL,
            file TEXT,
            line INTEGER,
            FOREIGN KEY(project_id)
                REFERENCES projects(id)
                ON DELETE CASCADE,
            FOREIGN KEY(source_symbol_id)
                REFERENCES symbols(id)
                ON DELETE CASCADE,
            FOREIGN KEY(target_symbol_id)
                REFERENCES symbols(id)
                ON DELETE SET NULL
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_references_project
        ON code_references(project_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_references_source
        ON code_references(source_symbol_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_references_target
        ON code_references(target_symbol_id)
    """)

    conn.commit()
    conn.close()


def create_project(name, path, mode):
    conn = get_connection()

    cursor = conn.execute(
        """
        INSERT INTO projects(name, path, mode)
        VALUES (?, ?, ?)
        """,
        (name, path, mode),
    )

    conn.commit()

    project_id = cursor.lastrowid

    conn.close()

    return project_id


def get_project(project_id):
    conn = get_connection()

    row = conn.execute(
        """
        SELECT *
        FROM projects
        WHERE id = ?
        """,
        (project_id,),
    ).fetchone()

    conn.close()

    return row


def list_projects():
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT *
        FROM projects
        ORDER BY id
        """
    ).fetchall()

    conn.close()

    return rows


def clear_project_code(project_id):
    conn = get_connection()

    conn.execute(
        """
        DELETE FROM code_references
        WHERE project_id = ?
        """,
        (project_id,),
    )

    conn.execute(
        """
        DELETE FROM symbols
        WHERE project_id = ?
        """,
        (project_id,),
    )

    conn.commit()
    conn.close()


def insert_symbol(
    project_id,
    file,
    name,
    qualified_name,
    symbol_type,
    language,
    parent_symbol_id,
    start_line,
    end_line,
    code,
):
    conn = get_connection()

    cursor = conn.execute(
        """
        INSERT INTO symbols (
            project_id,
            file,
            name,
            qualified_name,
            type,
            language,
            parent_symbol_id,
            start_line,
            end_line,
            code
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            file,
            name,
            qualified_name,
            symbol_type,
            language,
            parent_symbol_id,
            start_line,
            end_line,
            code,
        ),
    )

    symbol_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return symbol_id


def insert_reference(
    project_id,
    source_symbol_id,
    target_symbol_id,
    target_name,
    reference_type,
    file,
    line,
):
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO code_references (
            project_id,
            source_symbol_id,
            target_symbol_id,
            target_name,
            reference_type,
            file,
            line
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            source_symbol_id,
            target_symbol_id,
            target_name,
            reference_type,
            file,
            line,
        ),
    )

    conn.commit()
    conn.close()


def list_symbols(project_id):
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            s.*,
            p.qualified_name AS parent_qualified_name
        FROM symbols s
        LEFT JOIN symbols p
            ON p.id = s.parent_symbol_id
        WHERE s.project_id = ?
        ORDER BY s.file, s.start_line, s.id
        """,
        (project_id,),
    ).fetchall()

    conn.close()

    return rows


def get_symbol(project_id, symbol_name):
    conn = get_connection()

    row = conn.execute(
        """
        SELECT
            s.*,
            p.qualified_name AS parent_qualified_name
        FROM symbols s
        LEFT JOIN symbols p
            ON p.id = s.parent_symbol_id
        WHERE s.project_id = ?
          AND (
              s.name = ?
              OR s.qualified_name = ?
          )
        ORDER BY
            CASE
                WHEN s.qualified_name = ? THEN 0
                ELSE 1
            END,
            s.start_line
        LIMIT 1
        """,
        (
            project_id,
            symbol_name,
            symbol_name,
            symbol_name,
        ),
    ).fetchone()

    conn.close()

    return row

def get_symbol_by_id(project_id, symbol_id):
    conn = get_connection()

    row = conn.execute(
        """
        SELECT
            s.*,
            p.qualified_name AS parent_qualified_name
        FROM symbols s
        LEFT JOIN symbols p
            ON p.id = s.parent_symbol_id
        WHERE s.project_id = ?
          AND s.id = ?
        LIMIT 1
        """,
        (
            project_id,
            symbol_id,
        )
    ).fetchone()

    conn.close()

    return row

def get_callees(project_id, symbol_name):
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            r.*,

            s.name AS target_symbol,
            s.qualified_name AS target_qualified_name,
            s.file AS target_file,
            s.start_line AS target_start_line,
            s.end_line AS target_end_line,

            source.name AS source_symbol,
            source.qualified_name AS source_qualified_name

        FROM code_references r

        LEFT JOIN symbols s
            ON s.id = r.target_symbol_id

        LEFT JOIN symbols source
            ON source.id = r.source_symbol_id

        WHERE r.project_id = ?

          AND r.source_symbol_id = (
              SELECT id
              FROM symbols
              WHERE project_id = ?
                AND (
                    name = ?
                    OR qualified_name = ?
                )
              ORDER BY
                  CASE
                      WHEN qualified_name = ? THEN 0
                      ELSE 1
                  END,
                  id
              LIMIT 1
          )

        ORDER BY r.line
        """,
        (
            project_id,
            project_id,
            symbol_name,
            symbol_name,
            symbol_name,
        ),
    ).fetchall()

    conn.close()

    return rows


def get_callers(project_id, symbol_name):
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            r.*,

            s.name AS target_symbol,
            s.qualified_name AS target_qualified_name,

            source.name AS source_symbol,
            source.qualified_name AS source_qualified_name,
            source.file AS source_file,
            source.start_line AS source_start_line,
            source.end_line AS source_end_line

        FROM code_references r

        LEFT JOIN symbols s
            ON s.id = r.target_symbol_id

        LEFT JOIN symbols source
            ON source.id = r.source_symbol_id

        WHERE r.project_id = ?

          AND r.target_symbol_id = (
              SELECT id
              FROM symbols
              WHERE project_id = ?
                AND (
                    name = ?
                    OR qualified_name = ?
                )
              ORDER BY
                  CASE
                      WHEN qualified_name = ? THEN 0
                      ELSE 1
                  END,
                  id
              LIMIT 1
          )

        ORDER BY r.line
        """,
        (
            project_id,
            project_id,
            symbol_name,
            symbol_name,
            symbol_name,
        ),
    ).fetchall()

    conn.close()

    return rows
def get_children(project_id, parent_symbol_id):
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            s.*,
            p.qualified_name AS parent_qualified_name
        FROM symbols s
        LEFT JOIN symbols p
            ON p.id = s.parent_symbol_id
        WHERE s.project_id = ?
          AND s.parent_symbol_id = ?
        ORDER BY s.start_line
        """,
        (
            project_id,
            parent_symbol_id,
        )
    ).fetchall()

    conn.close()

    return rows
