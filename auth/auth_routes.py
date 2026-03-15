from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, UserMixin
from src.database import users_collection

auth = Blueprint("auth", __name__)
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.name = user_data.get("name", "")
        self.dob = user_data.get("dob", "")
        self.gender = user_data.get("gender", "")
        self.email = user_data.get("email", "")
        self.mobile = user_data.get("mobile", "")
    def get_id(self):
        return self.id

@auth.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            flash("Email already registered!", "danger")
            return redirect(url_for("auth.register"))
        hashed_password = generate_password_hash(password)
        users_collection.insert_one({
            "name": name,
            "email": email,
            "password": hashed_password
        })
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("auth.login"))
    return render_template("register.html")

@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            login_user(User(user))
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password", "danger")
            return render_template("login.html")
    return render_template("login.html")

@auth.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        new_password = request.form.get("new_password")
        confirm_password = request.form.get("confirm_password")
        user = users_collection.find_one({"email": email})
        if not user:
            flash("Email not registered!", "danger")
            return redirect(url_for("auth.forgot_password"))
        if new_password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("auth.forgot_password"))
        from werkzeug.security import generate_password_hash
        hashed_password = generate_password_hash(new_password)
        users_collection.update_one(
            {"email": email},
            {"$set": {"password": hashed_password}}
        )
        flash("Password reset successfully! Please login.", "success")
        return redirect(url_for("auth.login"))
    return render_template("forgot_password.html")

@auth.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))