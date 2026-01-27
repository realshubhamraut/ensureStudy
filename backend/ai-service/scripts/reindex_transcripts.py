#!/usr/bin/env python3
"""
Re-index existing meeting transcripts from MongoDB into Qdrant
Run this script to populate the vector database after collection reset
"""
import asyncio
import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor.motor_asyncio import AsyncIOMotorClient
from app.services.meeting_embedding_service import meeting_embedding_service

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB = os.getenv('MONGO_DB', 'ensure_study_meetings')


async def reindex_all_transcripts():
    """Re-index all existing transcripts from MongoDB into Qdrant"""
    print("=" * 60)
    print("Re-indexing Meeting Transcripts to Qdrant")
    print("=" * 60)
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DB]
    
    # Get all transcripts
    cursor = db.meeting_transcripts.find({})
    transcripts = await cursor.to_list(length=100)
    
    print(f"\nFound {len(transcripts)} transcripts in MongoDB")
    
    if not transcripts:
        print("No transcripts to index!")
        return
    
    total_chunks = 0
    
    for i, transcript in enumerate(transcripts, 1):
        recording_id = transcript.get('recording_id', 'unknown')
        meeting_id = transcript.get('meeting_id', 'unknown')
        classroom_id = transcript.get('classroom_id', 'unknown')
        
        print(f"\n[{i}/{len(transcripts)}] Indexing recording: {recording_id[:8]}...")
        
        # Get segments
        segments = transcript.get('segments', [])
        if not segments:
            # Try to create segments from full_text
            full_text = transcript.get('full_text', '')
            if full_text:
                # Create a single segment from full text
                segments = [{
                    'text': full_text,
                    'start': 0,
                    'end': transcript.get('duration_seconds', 0),
                    'speaker_id': 0,
                    'speaker_name': 'Teacher'
                }]
        
        if not segments:
            print(f"  ⚠ No segments found, skipping")
            continue
        
        # Convert segments to the format expected by embed_transcript
        formatted_segments = []
        for seg in segments:
            if isinstance(seg, dict):
                formatted_segments.append({
                    'text': seg.get('text', ''),
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'speaker_id': seg.get('speaker_id', 0),
                    'speaker_name': seg.get('speaker_name', 'Teacher')
                })
        
        if not formatted_segments:
            print(f"  ⚠ Could not format segments, skipping")
            continue
        
        try:
            # Embed and store
            chunks = await meeting_embedding_service.embed_transcript(
                recording_id=recording_id,
                meeting_id=meeting_id,
                classroom_id=classroom_id,
                segments=formatted_segments,
                meeting_title=f"Meeting {recording_id[:8]}"
            )
            print(f"  ✓ Indexed {chunks} chunks")
            total_chunks += chunks
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Re-indexing complete!")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"{'=' * 60}")
    
    client.close()


if __name__ == "__main__":
    asyncio.run(reindex_all_transcripts())
