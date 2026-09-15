from functools import wraps
from flask import session, redirect, url_for, flash


def login_user(user):
    session["user_id"] = user.id
    session["user_role"] = user.role
    session["user_name"] = user.fullname
    session["user_email"] = user.email


def logout_user():
    session.clear()


def current_user():
    return {
        "id": session.get("user_id"),
        "role": session.get("user_role"),
        "name": session.get("user_name"),
        "email": session.get("user_email"),
    }


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Bạn cần đăng nhập trước.")
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper


def admin_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Bạn cần đăng nhập trước.")
            return redirect(url_for("login"))
        if session.get("user_role") != "admin":
            flash("Bạn không có quyền truy cập khu vực quản trị.")
            return redirect(url_for("user_dashboard"))
        return func(*args, **kwargs)
    return wrapper

def admin_api_required(func):
    """Admin guard for JSON endpoints: never flash or redirect into user pages."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"success": False, "error": "authentication_required"}), 401
        if session.get("user_role") != "admin":
            return jsonify({"success": False, "error": "admin_required"}), 403
        return func(*args, **kwargs)
    return wrapper
