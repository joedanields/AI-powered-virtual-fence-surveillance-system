"""
Flask web dashboard — intrusion history viewer and live status.

Provides:
  • /                — dashboard home (recent events + stats)
  • /events          — paginated event list
  • /events/<id>     — event detail
  • /api/events      — JSON API for events
  • /api/stats       — daily intrusion statistics
  • /evidence/<file> — serve saved screenshots
"""

import os
import math
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify, send_from_directory, abort
)

from config import AppConfig, EVIDENCE_DIR
from database import IntrusionDatabase


def create_app(config: AppConfig) -> Flask:
    """Application factory — returns a configured Flask app."""

    app = Flask(
        __name__,
        template_folder=str(config.database.db_path).replace("data/intrusions.db", "") + "web/templates",
        static_folder=str(config.database.db_path).replace("data/intrusions.db", "") + "web/static",
    )
    app.secret_key = config.dashboard.secret_key

    # Shared database instance
    db = IntrusionDatabase(config.database)
    per_page = config.dashboard.events_per_page

    # ── Routes ─────────────────────────────────
    @app.route("/")
    def index():
        """Dashboard landing page."""
        recent = db.get_events(limit=10)
        total = db.count_events()
        stats = db.get_daily_stats(days=7)
        return render_template(
            "index.html",
            events=recent,
            total_events=total,
            daily_stats=stats,
            now=datetime.now(),
        )

    @app.route("/events")
    def events_list():
        """Paginated event list."""
        page = request.args.get("page", 1, type=int)
        camera = request.args.get("camera", None)
        status = request.args.get("status", None)
        offset = (page - 1) * per_page
        events = db.get_events(limit=per_page, offset=offset,
                               camera_id=camera, status=status)
        total = db.count_events(camera_id=camera)
        total_pages = max(1, math.ceil(total / per_page))
        return render_template(
            "events.html",
            events=events,
            page=page,
            total_pages=total_pages,
            total=total,
        )

    @app.route("/events/<int:event_id>")
    def event_detail(event_id):
        """Single event detail view."""
        event = db.get_event_by_id(event_id)
        if not event:
            abort(404)
        return render_template("event_detail.html", event=event)

    @app.route("/evidence/<path:filename>")
    def serve_evidence(filename):
        """Serve saved evidence files (screenshots, clips)."""
        return send_from_directory(str(EVIDENCE_DIR), filename)

    # ── JSON API ───────────────────────────────
    @app.route("/api/events")
    def api_events():
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)
        camera = request.args.get("camera", None)
        events = db.get_events(limit=limit, offset=offset, camera_id=camera)
        return jsonify({"events": events, "total": db.count_events(camera_id=camera)})

    @app.route("/api/stats")
    def api_stats():
        days = request.args.get("days", 7, type=int)
        stats = db.get_daily_stats(days=days)
        return jsonify({"stats": stats})

    @app.route("/api/events/<int:event_id>/acknowledge", methods=["POST"])
    def api_acknowledge(event_id):
        ok = db.update_status(event_id, "acknowledged")
        return jsonify({"success": ok})

    return app
