-- Migration: Add OCR bounding boxes column
-- Date: 2026-01-22
-- Description: Store line-level OCR data with bounding boxes for better text correlation

-- Add ocr_lines JSONB column to store line-level OCR data
-- Structure: [{"text": "...", "bbox": [x, y, w, h], "confidence": 0.92}, ...]
ALTER TABLE digitized_note_pages ADD COLUMN IF NOT EXISTS 
    ocr_lines JSONB DEFAULT '[]'::jsonb;

-- Add index for faster JSONB queries (if needed for search)
CREATE INDEX IF NOT EXISTS idx_note_pages_ocr_lines 
    ON digitized_note_pages USING gin (ocr_lines);

-- Update status enum to include new quality states
-- Note: PostgreSQL doesn't allow easy enum modification, so we'll use CHECK constraint
-- The status column already supports strings, so we just document the new values:
-- - 'pending': Initial state
-- - 'enhanced': Image enhancement complete
-- - 'ocr_done': OCR completed with high confidence (>=0.75)
-- - 'low_confidence': OCR completed but confidence is between 0.60-0.75
-- - 'needs_review': OCR completed but confidence < 0.60, needs human review
-- - 'embedded': Vector embeddings created

COMMENT ON COLUMN digitized_note_pages.ocr_lines IS 
    'Line-level OCR data with bounding boxes: [{"text": "...", "bbox": [x, y, w, h], "confidence": 0.92}]';

COMMENT ON COLUMN digitized_note_pages.status IS 
    'Processing status: pending, enhanced, ocr_done (>=0.75 conf), low_confidence (0.60-0.75), needs_review (<0.60), embedded';
