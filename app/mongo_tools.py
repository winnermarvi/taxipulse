"""MongoDB access for the Control Center (the NoSQL serving layer).

Queries are defined here in Python AND shown in the UI as their mongosh
equivalent, so the examiner sees exactly what is being executed.
"""
from __future__ import annotations

from taxipulse import config


def client(timeout_ms: int = 1500):
    from pymongo import MongoClient
    return MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=timeout_ms)


def ping() -> tuple[bool, str]:
    try:
        c = client()
        info = c.server_info()
        return True, info.get("version", "?")
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


def db():
    return client()[config.MONGO_DB]


def collection_stats() -> list[dict]:
    d = db()
    return sorted(({"collection": n, "documents": d[n].estimated_document_count(),
                    "indexes": len(d[n].index_information())} for n in d.list_collection_names()),
                  key=lambda r: -r["documents"])


def overview() -> dict | None:
    return db().overview.find_one({}, {"_id": 0})


BOROUGH_PIPELINE = [
    {"$group": {"_id": "$borough", "pickups": {"$sum": "$pickup_trips"}, "zones": {"$sum": 1},
                "avg_fare": {"$avg": "$avg_fare"}}},
    {"$sort": {"pickups": -1}},
]
BOROUGH_MONGOSH = """db.zone_metrics.aggregate([
  { $group: { _id: "$borough",
              pickups:  { $sum: "$pickup_trips" },
              zones:    { $sum: 1 },
              avg_fare: { $avg: "$avg_fare" } } },
  { $sort: { pickups: -1 } }
])"""


def borough_aggregation() -> list[dict]:
    return list(db().zone_metrics.aggregate(BOROUGH_PIPELINE))


def top_routes(n: int = 10) -> list[dict]:
    return list(db().route_metrics.find({}, {"_id": 0, "pu_zone": 1, "do_zone": 1, "trip_count": 1, "avg_fare": 1,
                                              "avg_distance_mi": 1}).sort("trip_count", -1).limit(n))


def _stages(node, out):
    if isinstance(node, dict):
        if "stage" in node:
            out.append(node["stage"])
        for k in ("inputStage", "inputStages", "queryPlan"):
            if k in node:
                v = node[k]
                for x in (v if isinstance(v, list) else [v]):
                    _stages(x, out)
    return out


def explain_lookup(zone_id: int, date: str, force_collscan: bool = False) -> dict:
    """Run the same query with the index and (optionally) forced to a full collection scan."""
    find = {"find": "hourly_zone_demand", "filter": {"zone_id": zone_id, "pickup_date": date}}
    if force_collscan:
        find["hint"] = {"$natural": 1}
    res = db().command({"explain": find, "verbosity": "executionStats"})
    es = res["executionStats"]
    return {
        "nReturned": es["nReturned"],
        "totalDocsExamined": es["totalDocsExamined"],
        "totalKeysExamined": es["totalKeysExamined"],
        "executionTimeMillis": es["executionTimeMillis"],
        "stages": " → ".join(_stages(es.get("executionStages", {}), [])) or "?",
        "collection_size": db().hourly_zone_demand.estimated_document_count(),
    }


def explain_mongosh(zone_id: int, date: str, force_collscan: bool = False) -> str:
    hint = '\n  .hint({ $natural: 1 })        // force a full collection scan' if force_collscan else ""
    return (f'db.hourly_zone_demand\n  .find({{ zone_id: {zone_id}, pickup_date: "{date}" }}){hint}\n'
            f'  .explain("executionStats").executionStats')
