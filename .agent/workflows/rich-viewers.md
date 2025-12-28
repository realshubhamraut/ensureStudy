---
description: how to implement rich viewers and hover overlays for the AI Tutor sidebar
---

# Rich Viewer Implementation Workflow

## Overview
Enhance the AI Tutor sidebar with rich viewers for all content types and hover overlays for quick actions.

## Phase 1: Hover Overlays for All Resource Types
1. Create a reusable `HoverOverlay` component that shows on hover
2. Add "Open" button that appears on hover for:
   - YouTube video thumbnails
   - Document items
   - Image items  
   - Website/article items
3. Style with gentle fade-in animation

## Phase 2: YouTube Thumbnail Enhancement
1. Make video thumbnails larger/full-width in the list
2. Add play button overlay
3. Add duration badge

## Phase 3: Image Viewer with Zoom Controls
1. Add zoom in/out buttons (+/-)
2. Add zoom percentage display
3. Add reset zoom button
4. Enable mouse wheel zoom
5. Enable pan when zoomed

## Phase 4: Rich PDF Viewer
1. Install `react-pdf` library
2. Create PDF viewer component with:
   - Page navigation
   - Zoom controls
   - Fullscreen option
   - Page count display

## Phase 5: PPTX Viewer
1. Use `pptx-viewer` or render as image slides
2. Add slide navigation
3. Add fullscreen presentation mode

## Phase 6: Mindmap Renderer (Mermaid)
1. Install `mermaid` library
2. Create Mindmap component
3. Parse mermaid code from backend
4. Render interactive mindmap
5. Add zoom/pan controls

## Dependencies to Install
```bash
npm install react-pdf mermaid
npm install @types/mermaid --save-dev
```

## Files to Modify
- `frontend/app/(dashboard)/chat/page.tsx` - Main viewer components
- Create new component files for each viewer type
