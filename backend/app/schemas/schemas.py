import pathway as pw


class DetectionSchema(pw.Schema):
    timestamp:      pw.DateTimeUtc
    violation_type: str
    confidence:     float
    zone:           str
    camera_id:      str