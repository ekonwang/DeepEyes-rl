import os
import time
import json
import random
import multiprocessing as mp

# --- 环境：强制使用 SQLite 缓存（若你想测 Redis，注释掉下一行并设置 REDIS_URL）---
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("SEARCH_CACHE_SQLITE_PATH", "search_cache.sqlite")
os.environ.setdefault("SEARCH_CACHE_TTL_SEC", "21600000")
os.environ.setdefault("SEARCH_CACHE_MAXSIZE", "500000")

# --- 你的真实实现应已可导入：search_tool_core / _make_key / init_search_cache 等 ---
# 为了脚本可单独跑，这里做“兜底”定义：若找不到 run_search / search_tool_core，就注入一个最小可运行版本。
try:
    from your_module import search_tool_core  # 如果你已有模块，就用真实的
except Exception:
    # ======= 下面是最小可运行的“假实现”，便于独立跑通 =======
    # 缓存：使用你之前给的 SQLiteCache/RedisCache 之一。这里内嵌一个最简 SQLiteCache（WAL + TTL）。
    import sqlite3
    from threading import RLock

    def _dumps(obj): return json.dumps(obj, ensure_ascii=False).encode("utf-8")
    def _loads(buf): return json.loads(buf.decode("utf-8"))

    class SQLiteCache:
        def __init__(self, path="search_cache.sqlite", ttl_sec=21600, maxsize=5000, busy_timeout_ms=3000):
            self.path, self.ttl, self.maxsize = path, ttl_sec, maxsize
            self.busy_timeout_ms = busy_timeout_ms
            self._lock = RLock()
            self._init_db()
        def _connect(self):
            conn = sqlite3.connect(self.path, timeout=self.busy_timeout_ms/1000, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
            return conn
        def _init_db(self):
            with self._connect() as c:
                c.execute("""CREATE TABLE IF NOT EXISTS cache(
                    k TEXT PRIMARY KEY, v BLOB NOT NULL, ts REAL NOT NULL, last_access_ts REAL NOT NULL)""")
        def get(self, key):
            now = time.time()
            with self._lock, self._connect() as c:
                row = c.execute("SELECT v, ts FROM cache WHERE k=?", (key,)).fetchone()
                if not row: return None
                v, ts = row
                if now - ts > self.ttl:
                    c.execute("DELETE FROM cache WHERE k=?", (key,)); return None
                c.execute("UPDATE cache SET last_access_ts=? WHERE k=?", (now, key))
                return _loads(v)
        def set(self, key, value):
            now = time.time()
            with self._lock, self._connect() as c:
                c.execute("BEGIN IMMEDIATE;")
                c.execute("""INSERT INTO cache(k,v,ts,last_access_ts) VALUES(?,?,?,?)
                             ON CONFLICT(k) DO UPDATE SET v=excluded.v, ts=excluded.ts, last_access_ts=excluded.last_access_ts
                          """, (key, _dumps(value), now, now))
                # 轻量清理
                c.execute("DELETE FROM cache WHERE (? - ts) > ?", (now, self.ttl))
                # 近似 LRU
                cnt = c.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                if cnt > self.maxsize:
                    c.execute("""DELETE FROM cache WHERE k IN (
                        SELECT k FROM cache ORDER BY last_access_ts ASC LIMIT ?
                    )""", (cnt - self.maxsize,))
                c.execute("COMMIT;")

    _CACHE = SQLiteCache(
        path=os.environ.get("SEARCH_CACHE_SQLITE_PATH", "search_cache.sqlite"),
        ttl_sec=int(os.environ.get("SEARCH_CACHE_TTL_SEC", "21600")),
        maxsize=int(os.environ.get("SEARCH_CACHE_MAXSIZE", "5000")),
    )

    import re
    _space_re = re.compile(r"\s+")
    def _normalize_query(q: str) -> str:
        return _space_re.sub(" ", q.strip()).casefold()
    def _make_key(query: str, retriever_name: str) -> str:
        return json.dumps({"q": _normalize_query(query), "retriever": retriever_name}, sort_keys=True, ensure_ascii=False)

    # 假 run_search：模拟 100~300ms 的网络延迟与返回
    def run_search(query: str, retriever_name: str = "tavily"):
        time.sleep(random.uniform(0.10, 0.30))
        return [{"title": f"Result for {query}", "url": "https://example.com", "retriever": retriever_name}]

    def _log_kv(_ok, msg, kv):  # 极简日志
        print(f"[{time.strftime('%H:%M:%S')}] {msg} | {kv}")

    def search_tool_core(query: str, debug: bool = False, retriever_name: str = 'tavily'):
        key = _make_key(query, retriever_name)
        try:
            cached = _CACHE.get(key)
            if cached is not None:
                if debug: _log_kv(True, "Search cache hit (persistent)", {"retriever": retriever_name})
                return {"success": True, "query": query, "results": cached}
            if debug: _log_kv(True, "Executing web search", {"query": query, "retriever": retriever_name})
            results = run_search(query, retriever_name=retriever_name)
            _CACHE.set(key, results)
            return {"success": True, "query": query, "results": results}
        except Exception as e:
            raise Exception(f"Search tool execution failed: {str(e)}")

def _worker(args):
    idx, query, retriever, debug = args
    t0 = time.perf_counter()
    ok, n = False, "n/a"
    err = ""
    try:
        res = search_tool_core(query, debug=debug, retriever_name=retriever)
        ok = bool(res.get("success"))
        r = res.get("results")
        n = (len(r) if isinstance(r, list) else "n/a")
    except Exception as e:
        err = str(e)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {"i": idx, "pid": os.getpid(), "ms": elapsed_ms, "ok": ok, "n": n, "err": err}

if __name__ == "__main__":
    # Windows/macOS/Linux 跨平台更稳：spawn
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    PROCS = 64
    QUERY = "concurrency-cache test query"
    RETRIEVER = "tavily"
    DEBUG = False  # 如需看到 cache 命中/未命中日志可设 True

    print(f"Spawning {PROCS} processes …")
    args = [(i, QUERY, RETRIEVER, DEBUG) for i in range(PROCS)]

    t0 = time.perf_counter()
    with mp.Pool(processes=PROCS) as pool:
        results = pool.map(_worker, args)
    total_ms = int((time.perf_counter() - t0) * 1000)

    # 按索引排序，打印每个进程的耗时
    results.sort(key=lambda x: x["i"])
    print("\nidx\tpid\tms\tok\tresults\terror")
    for r in results:
        print(f"{r['i']:02d}\t{r['pid']}\t{r['ms']}\t{r['ok']}\t{r['n']}\t{r['err'][:60]}")

    print(f"\nTotal wall time: {total_ms} ms")
