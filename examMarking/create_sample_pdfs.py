"""
Utility script to create sample student answer PDFs for testing
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from pathlib import Path

def create_sample_pdf(filename, content, title="Student Answer"):
    """Create a PDF with the given content"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, height - inch, title)
    
    # Content
    c.setFont("Helvetica", 11)
    text_object = c.beginText(inch, height - 1.5 * inch)
    text_object.setFont("Helvetica", 11)
    text_object.setLeading(14)
    
    # Wrap text
    for paragraph in content.split('\n\n'):
        for line in paragraph.split('\n'):
            text_object.textLine(line)
        text_object.textLine("")  # Empty line between paragraphs
    
    c.drawText(text_object)
    c.save()
    print(f"Created: {filename}")

# Sample answers for Q1 (BST) with varying quality

# Excellent answer (90-100%)
excellent_q1 = """Binary Search Tree is a tree data structure where each node has maximum two children called left and right child. The main property is that left subtree contains values smaller than the node and right subtree contains larger values.

For insertion, we start from root and compare the value. If smaller go left, if larger go right, until we find empty position. Time complexity is O(log n) for balanced tree.

For deletion, three cases exist:
1. Leaf node - simply remove it
2. One child - replace node with its child
3. Two children - find inorder successor from right subtree, copy value, then delete successor

BST allows efficient searching, insertion and deletion operations."""

# Good answer (75-85%)
good_q1 = """Binary Search Tree is a tree where left child is smaller and right child is larger than parent node. This makes searching faster.

Insertion: Compare new value with root, go left if smaller or right if bigger. Keep comparing until empty spot found.

Deletion has different cases. If node has no children, just delete it. If one child exists, replace with that child. If two children, use successor or predecessor method.

BST operations are efficient compared to linear structures."""

# Average answer (60-70%)
average_q1 = """Binary tree is a data structure with nodes connected in hierarchy. Each node can have left and right children.

To insert, we compare values and place smaller values on left and bigger on right side.

To delete a node, we need to handle different situations depending on number of children the node has."""

# Weak answer (40-50%)
weak_q1 = """A tree is a data structure made of nodes. Binary trees have two children per node.

We can add new nodes by comparing values. We can also remove nodes from the tree.

Trees are used for storing data in organized way."""

# Output directory
output_dir = Path("data/sample_student_answers")
output_dir.mkdir(parents=True, exist_ok=True)

# Create PDFs
create_sample_pdf(
    str(output_dir / "Q1_Student_Excellent.pdf"),
    excellent_q1,
    "Question 1: Binary Search Trees - Student Answer"
)

create_sample_pdf(
    str(output_dir / "Q1_Student_Good.pdf"),
    good_q1,
    "Question 1: Binary Search Trees - Student Answer"
)

create_sample_pdf(
    str(output_dir / "Q1_Student_Average.pdf"),
    average_q1,
    "Question 1: Binary Search Trees - Student Answer"
)

create_sample_pdf(
    str(output_dir / "Q1_Student_Weak.pdf"),
    weak_q1,
    "Question 1: Binary Search Trees - Student Answer"
)

print("\nAll sample PDFs created successfully!")
