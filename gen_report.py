import json
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF, XPos, YPos
import os

# Load the JSON data
with open('processed_results.json') as f:
    data = json.load(f)

# Create directory for gauge images
if not os.path.exists('gauges'):
    os.makedirs('gauges')

# Function to create half-circle gauge chart
def create_half_gauge(value, label, filename):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(2, 1))
    
    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)
    
    # Color selection
    if value >= 0:
        color = 'green'
    else:
        color = 'red'
    
    ax.fill_between(theta, 0, r, color='lightgrey')
    ax.fill_between(theta, 0, r * (value + 1) / 2, color=color)
    
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_theta_zero_location('S')
    ax.set_theta_direction(-1)
    
    ax.text(0, -0.4, f"{label}\n{value:.2f}", ha='center', va='center', fontsize=12, weight='bold')
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# Create PDF
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Sentiment Analysis Report', border=False, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    def chapter_title(self, chunk_reference):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, f'Chunk Reference: {chunk_reference}', border=False, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)

    def chapter_body(self, summary, commentary):
        self.set_font('Helvetica', '', 12)
        self.multi_cell(0, 10, f"Summary:\n{summary}")
        self.ln(5)
        self.multi_cell(0, 10, f"Commentary:\n{commentary}")
        self.ln(10)

    def add_gauge(self, polarity_image, subjectivity_image):
        y = self.get_y()
        self.image(polarity_image, 10, y, 95, 50)
        self.image(subjectivity_image, 105, y, 95, 50)
        self.ln(55)

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Process each chunk and add to PDF
for chunk in data:
    chunk_ref = chunk['chunk_reference']
    summary = chunk['summary']
    commentary = chunk['commentary']
    polarity = chunk['sentiment_polarity']
    subjectivity = chunk['sentiment_subjectivity']

    # Create gauge images
    polarity_image = f'gauges/polarity_{chunk_ref}.png'
    subjectivity_image = f'gauges/subjectivity_{chunk_ref}.png'
    create_half_gauge(polarity, 'Polarity', polarity_image)
    create_half_gauge(subjectivity, 'Subjectivity', subjectivity_image)

    pdf.chapter_title(chunk_ref)
    pdf.add_gauge(polarity_image, subjectivity_image)
    pdf.chapter_body(summary, commentary)

pdf.output('report.pdf')

# Clean up gauge images
for file in os.listdir('gauges'):
    os.remove(os.path.join('gauges', file))
os.rmdir('gauges')
