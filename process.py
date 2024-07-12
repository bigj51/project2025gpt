from PyPDF2 import PdfReader
import openai
from textblob import TextBlob
import json
import keys

# Set up the OpenAI API key
openai.api_key = keys.OPENAI_API_KEY

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text(text, max_tokens=32000):
    words = text.split()
    chunks = []
    chunk = []
    chunk_size = 0

    for word in words:
        word_size = len(word) + 1  # Add 1 for the space
        if chunk_size + word_size > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
            chunk_size = 0
        chunk.append(word)
        chunk_size += word_size

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

# Function to get summarization, sentiment, and commentary from GPT
def process_chunk(chunk):
    MODEL = 'gpt-4o'

    # summary
    summary_response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Please summarize the following text:\n\n{chunk}",
            }
        ]
    )
    summary = (summary_response.choices)[0].message.content.strip()

    # sentiment
    blob = TextBlob(chunk)
    sentiment = blob.sentiment

    # general commentary
    commentary_response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Please provide a general commentary on the following text:\n\n{chunk}",
            }
        ]
    )
    
    commentary = (commentary_response.choices)[0].message.content.strip()

    # democrat commentary
    d_commentary_response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Please provide democratic party type commentary on the following text:\n\n{chunk}",
            }
        ]
    )
    
    d_commentary = (d_commentary_response.choices)[0].message.content.strip()

    # republican commentary
    r_commentary_response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Please provide republican party type commentary on the following text:\n\n{chunk}",
            }
        ]
    )
    r_commentary = (r_commentary_response.choices)[0].message.content.strip()

    return summary, sentiment, commentary, d_commentary, r_commentary

# Main function to process PDF
def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)

    i=0
    # write the chunks to a file with a refernce number.  to minimze the file sizes
    with open('chunk_ref.json', 'a') as f:
        for chunk in chunks:
            summary, sentiment, commentary, d_commentary, r_commentary = process_chunk(chunk)
            chunk_data= {
                "chunk": chunk,
                "chunk_reference": i
            }
            #json.dump(chunk_data, f, ensure_ascii=False, indent=4) #this is annoying.  since we have the chunk size we should be able to recreate the chunks, currently 32000

            processed_data={
                "chunk_reference": i,
                "summary": summary,
                "sentiment_polarity": sentiment.polarity,
                "sentiment_subjectivity": sentiment.subjectivity,
                "commentary": commentary,
                "d_commentary": d_commentary,
                "r_commentary": r_commentary,
            }

            # write a smaller version without the chunks
            with open('processed_results.json', 'a') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=4)
            
            # done
            print(f"::::chunk {i}/{len(chunks)} processed::::")
            i+=1
    return 

# Usage example
pdf_path = "2025_MandateForLeadership_FULL.pdf"
process_pdf(pdf_path)
print("Processing complete.")
