// MongoDB Initialization Script
// Run on first container start

// Switch to the application database
db = db.getSiblingDB('ensure_study_meetings');

// Create collections with validators
db.createCollection('meetings', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['meeting_id', 'title', 'host_id'],
            properties: {
                meeting_id: { bsonType: 'string' },
                title: { bsonType: 'string' },
                classroom_id: { bsonType: 'string' },
                host_id: { bsonType: 'string' },
                status: { enum: ['scheduled', 'active', 'completed', 'cancelled'] }
            }
        }
    }
});

db.createCollection('transcripts');
db.createCollection('summaries');
db.createCollection('meeting_qa');
db.createCollection('meeting_recordings');

// Create indexes
db.meetings.createIndex({ "meeting_id": 1 }, { unique: true });
db.meetings.createIndex({ "classroom_id": 1 });
db.meetings.createIndex({ "host_id": 1 });
db.meetings.createIndex({ "scheduled_at": 1 });
db.meetings.createIndex({ "status": 1 });

db.transcripts.createIndex({ "meeting_id": 1 }, { unique: true });
db.transcripts.createIndex({ "$**": "text" });

db.summaries.createIndex({ "meeting_id": 1 });
db.summaries.createIndex({ "summary_type": 1 });

db.meeting_qa.createIndex({ "meeting_id": 1 });

db.meeting_recordings.createIndex({ "meeting_id": 1 });
db.meeting_recordings.createIndex({ "processing_status": 1 });

print('✅ MongoDB initialized for ensure_study_meetings');
