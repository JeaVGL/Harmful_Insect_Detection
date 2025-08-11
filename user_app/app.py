
from flask import Flask, request, jsonify, send_file
import sqlite3
from datetime import datetime
import csv
import io

app = Flask(__name__)

DANGEROUS_SPECIES = ["Spodoptera litura", "Bollworm", "Rice planthopper"]
ALERT_THRESHOLD = 20  # if detected more than X times

def init_db():
    conn = sqlite3.connect('pest_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (timestamp TEXT, class_name TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return "<h2>Insect Detection API</h2><p>Endpoints: /upload, /stats, /history, /export</p>"

@app.route('/upload', methods=['POST'])
def upload():
    data = request.json
    if not data or "class" not in data or "confidence" not in data:
        return jsonify({"error": "Invalid data"}), 400

    class_name = data["class"]
    confidence = float(data["confidence"])
    timestamp = datetime.now().isoformat()

    conn = sqlite3.connect('pest_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO detections VALUES (?, ?, ?)", (timestamp, class_name, confidence))
    conn.commit()
    conn.close()

    return jsonify({"status": "success"}), 200

@app.route('/stats', methods=['GET'])
def stats():
    conn = sqlite3.connect('pest_data.db')
    c = conn.cursor()
    c.execute("SELECT class_name, COUNT(*) FROM detections GROUP BY class_name")
    rows = c.fetchall()
    conn.close()

    class_counts = {r[0]: r[1] for r in rows}
    alerts = [name for name in class_counts if name in DANGEROUS_SPECIES and class_counts[name] >= ALERT_THRESHOLD]

    return jsonify({
        "class_counts": class_counts,
        "alerts": alerts
    })

@app.route('/history', methods=['GET'])
def history():
    conn = sqlite3.connect('pest_data.db')
    c = conn.cursor()
    c.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 100")
    rows = c.fetchall()
    conn.close()

    return jsonify([
        {"timestamp": r[0], "class": r[1], "confidence": r[2]} for r in rows
    ])

@app.route('/export', methods=['GET'])
def export():
    conn = sqlite3.connect('pest_data.db')
    c = conn.cursor()
    c.execute("SELECT * FROM detections")
    rows = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "class", "confidence"])
    writer.writerows(rows)

    output.seek(0)
    return send_file(
        io.BytesIO(output.read().encode()),
        mimetype='text/csv',
        download_name='detections_export.csv',
        as_attachment=True
    )

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)
