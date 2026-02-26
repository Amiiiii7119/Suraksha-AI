Suraksha AI

AI-Driven Real-Time Construction Risk Intelligence Platform

Vision

Suraksha AI transforms traditional workplace supervision into a proactive, AI-powered risk intelligence system.
Instead of reacting to accidents, it continuously detects hazards, computes risk levels in real time, and provides structured safety analytics.

Built for high-risk industrial environments, the system demonstrates scalable real-time safety automation.

Problem

Construction sites face:

Delayed hazard detection

Manual and inconsistent monitoring

Lack of quantifiable risk metrics

No centralized safety intelligence system

This leads to preventable accidents and operational inefficiencies.

Solution

Suraksha AI integrates AI detection, real-time stream processing, and backend risk modeling into a unified architecture.

The Platform:

Detects safety violations:

Fire

Smoke

No Hardhat

No Safety Vest

No Mask

Streams events to a real-time processing engine (Pathway 0.29.0)

Computes dynamic weighted risk scores using deterministic logic

Stores structured event data in PostgreSQL

Displays live safety metrics and leaderboard analytics via an HTML dashboard

Technical Architecture

HTML Dashboard
⬇
FastAPI REST Backend
⬇
Pathway Stream Processing Engine
⬇
PostgreSQL Database

Engineering Highlights

Real-time stream-based risk computation

Deterministic weighted risk scoring model

Sliding-window event aggregation

Modular backend architecture

Secure environment variable handling

One-command system startup (run.sh)

Health monitoring endpoints

Production-oriented structure

Tech Stack

Python

FastAPI

Pathway 0.29.0

PostgreSQL

HTML Dashboard

YOLO-based detection logic

How to Run

From project root:

bash run.sh

This will automatically:

Start FastAPI backend

Initialize real-time risk engine

Launch frontend server

Open dashboard in browser

Access URLs

Frontend:
http://localhost:3000/app.html

Backend API Docs:
http://localhost:8000/docs

Default Login

Email:
amteshwarrajsingh@gmail.com

Password:
admin123

Impact

Suraksha AI enables:

Proactive accident prevention

Quantifiable safety intelligence

Data-driven compliance monitoring

Scalable deployment across multiple sites

Future Expansion

Live CCTV integration

Real-time alert notifications (SMS/Email)

Risk heatmaps

Enterprise-grade role management

Cloud-native deployment
