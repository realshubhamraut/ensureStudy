-- mermaid.lua - Pandoc Lua filter to render Mermaid diagrams using mermaid.ink
-- Uses mermaid.ink which accepts base64 encoded diagrams (no compression needed)

function CodeBlock(block)
    if block.classes[1] == "mermaid" then
        -- Base64 encode the mermaid code
        local encoded = pandoc.pipe("base64", {"-w", "0"}, block.text)
        -- Clean up the encoding (remove newlines)
        encoded = encoded:gsub("\n", "")
        
        -- mermaid.ink accepts plain base64
        local img_url = "https://mermaid.ink/img/" .. encoded
        
        -- Return the image
        return pandoc.Para({pandoc.Image({pandoc.Str("Diagram")}, img_url, "Mermaid Diagram")})
    end
end
