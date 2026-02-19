import sqlite3
import datetime

DB_FILE = "agent_memory.db"


class MemoryManager:
    def __init__(self):
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(DB_FILE)
        # ⚡ OPTIMIZATION: Enable Write-Ahead Logging for concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS skills 
                     (name TEXT PRIMARY KEY, code TEXT, description TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS history 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        try:
            c.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_index USING fts5(content, meta_info)''')
        except Exception:
            pass
        conn.commit()
        conn.close()

    def save_interaction(self, role, content):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("INSERT INTO history (role, content) VALUES (?, ?)", (role, str(content)))
        if len(content) > 10:
            meta = f"role:{role}|time:{datetime.datetime.now().isoformat()}"
            c.execute("INSERT INTO knowledge_index (content, meta_info) VALUES (?, ?)", (content, meta))
        conn.commit()
        conn.close()

    def retrieve_relevant_context(self, current_query, limit=2):  # Reduced limit for speed
        conn = self._get_conn()
        c = conn.cursor()

        # FTS Search
        clean_query = "".join([x if x.isalnum() else " " for x in current_query])
        search_terms = " OR ".join([f'"{term}"' for term in clean_query.split() if len(term) > 3])

        context_results = []
        if search_terms:
            try:
                # ⚡ OPTIMIZATION: Limit FTS results strictly
                c.execute(
                    f"SELECT content FROM knowledge_index WHERE knowledge_index MATCH ? ORDER BY rank LIMIT {limit}",
                    (search_terms,))
                for r in c.fetchall():
                    context_results.append(f"[MEMORY]: {r[0]}")
            except Exception:
                pass

        # Short term history (Last 5 is usually enough)
        c.execute("SELECT role, content FROM history ORDER BY id DESC LIMIT 5")
        recent_rows = c.fetchall()
        recent_context = [f"[{row[0].upper()}]: {row[1]}" for row in reversed(recent_rows)]

        conn.close()
        return "\n".join(context_results + ["\n--- RECENT CHAT ---"] + recent_context)

    # ... (Rest of Tool DB methods remain same, but replace `sqlite3.connect` with `self._get_conn()`) ...
    def save_skill(self, name, code, description):
        try:
            conn = self._get_conn()
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO skills (name, code, description) VALUES (?, ?, ?)",
                      (name, code, description))
            conn.commit()
            conn.close()
            return f"✅ Skill '{name}' saved."
        except Exception as e:
            return f"DB Error: {e}"

    def list_skills(self):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT name, description FROM skills")
        rows = c.fetchall()
        conn.close()
        return rows

    def get_tool_code(self, name):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT code FROM skills WHERE name = ?", (name,))
        res = c.fetchone()
        conn.close()
        return res[0] if res else None

    def delete_skill(self, name):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM skills WHERE name=?", (name,))
        conn.commit()
        conn.close()
        return f"Deleted {name}"


memory = MemoryManager()