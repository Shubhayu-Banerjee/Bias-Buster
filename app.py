import os
import requests
from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
import json
import re

# --- Configuration ---
# IMPORTANT: Replace with your actual Groq API Key
GROQ_API_KEY = "Your Key"

app = Flask(__name__)

# --- Dummy/Prototype Trusted Sources (Expand this significantly for production) ---
# IMPORTANT: Use lowercase keys for easier matching with AI suggestions
TRUSTED_SOURCE_LINKS = {
    "fact-checking organizations": [
        {"name": "Snopes", "url": "https://www.snopes.com/"},
        {"name": "PolitiFact", "url": "https://www.politifact.com/"},
        {"name": "Alt News (India)", "url": "https://www.altnews.in/"},
        {"name": "Boom Live (India)", "url": "https://www.boomlive.in/"}
    ],
    "academic research on language and education": [
        {"name": "JSTOR (Academic Articles)", "url": "https://www.jstor.org/"},
        {"name": "Google Scholar (Academic Search)", "url": "https://scholar.google.com/"},
        {"name": "ResearchGate (Academic Papers)", "url": "https://www.researchgate.net/"}
    ],
    "official government statistics on education": [
        {"name": "UNESCO Education Data", "url": "https://data.uis.unesco.org/"},
        {"name": "Ministry of Education (India)", "url": "https://www.education.gov.in/"},
        {"name": "National Sample Survey Office (NSSO) - India", "url": "https://mospi.gov.in/web/mospi/nsso"}
    ],
    "historical accounts of the mughal empire": [
        {"name": "Britannica: Mughal Empire", "url": "https://www.britannica.com/place/Mughal-Empire"},
        {"name": "Metropolitan Museum of Art: Mughal Dynasty",
         "url": "https://www.metmuseum.org/toah/hd/mugh/hd_mugh.htm"}
    ],
    "academic research on education and social inequality": [
        {"name": "National Bureau of Economic Research (NBER)",
         "url": "https://www.nber.org/papers/tags/education-and-social-inequality"},
        {"name": "Poverty & Race Research Action Council", "url": "https://prrac.org/"}
    ],
    "reputable news organizations": [
        {"name": "Reuters", "url": "https://www.reuters.com/"},
        {"name": "Associated Press (AP)", "url": "https://apnews.com/"},
        {"name": "The Hindu", "url": "https://www.thehindu.com/"},
        {"name": "BBC News", "url": "https://www.bbc.com/news/world/asia/india"},
        {"name": "The New York Times", "url": "https://www.nytimes.com/"},
        {"name": "The Wall Street Journal", "url": "https://www.wsj.com/"}
    ],
    "economic analysis and data": [
        {"name": "Reserve Bank of India (RBI)", "url": "https://www.rbi.org.in/"},
        {"name": "World Bank Data", "url": "https://data.worldbank.org/"},
        {"name": "International Monetary Fund (IMF)", "url": "https://www.imf.org/en/Data"}
    ],
    "climate science and data": [
        {"name": "NASA Climate", "url": "https://climate.nasa.gov/"},
        {"name": "IPCC (Intergovernmental Panel on Climate Change)", "url": "https://www.ipcc.ch/"},
        {"name": "NOAA Climate.gov", "url": "https://www.climate.gov/"}
    ]
}


# --- New Function to Clean JSON String ---
def clean_json_string(json_str):
    """
    Removes trailing commas from JSON strings, including those followed by newlines and spaces.
    This is crucial for LLMs that might incorrectly add them, making the JSON invalid.
    """
    # This regex is more robust:
    # It looks for a comma, followed by any whitespace (including newlines),
    # then immediately followed by a closing square bracket ']' or curly brace '}'.
    # It replaces the comma and whitespace with just the closing bracket/brace.
    cleaned_str = re.sub(r',\s*([\]}])', r'\1', json_str)

    # Also, address potential trailing commas before the very end of the string
    # where the AI might just end an object/array abruptly with a comma.
    # E.g., {"a":1,"b":2,} or [1,2,3,]
    cleaned_str = re.sub(r',\s*([}\]])\s*$', r'\1', cleaned_str.strip())  # strip to catch if it's at the very end

    return cleaned_str


# --- AI Helper Function (Adapted for Groq API) ---
def ask_ai(prompt, article_text):
    """
    Sends a prompt and article text to the Groq API for analysis.
    Instructs the AI to return a structured JSON response.
    """
    if not GROQ_API_KEY:
        return json.dumps({"error": "GROQ_API_KEY is not set."})

    # Truncate article_text to fit within model context window (approx 8192 tokens for llama3-8b-8192)
    # A token is roughly 4 characters, so 8000 tokens ~ 32000 characters
    # We'll take slightly less to leave room for the prompt and response.
    max_text_length = 30000  # Characters
    if len(article_text) > max_text_length:
        article_text = article_text[:max_text_length] + " [TRUNCATED FOR AI ANALYSIS]"

    messages = [
        {
            "role": "system",
            "content": """You are an expert at identifying misinformation patterns in text.
            Your primary goal is to identify rhetorical devices and logical fallacies that are used to *manipulate*, *deceive*, or *mislead* the reader.
            It is crucial to differentiate between genuinely manipulative tactics and strong, critical, or opinionated statements that are still fair or express legitimate concerns.
            Not every persuasive technique is a misinformation pattern. Focus on instances where the author intentionally distorts information, uses flawed logic, or appeals to emotion in a way that undermines factual accuracy or balanced understanding.

            Your entire response MUST be a **single, valid JSON object**. This object MUST contain three keys:
            1. 'overall_summary': A JSON array of strings, where each string is a bullet point summarizing the overall bias, tone, and legitimacy of the article. If the article is largely fair, state that.
            2. 'overall_sentiment': (string) A single word indicating the overall sentiment/flag for the article based on the detected patterns and summary. Choose from 'Read-Freely' (mostly fair/minor issues, suitable for general consumption), 'Read Carefully' (some moderate issues, potentially misleading), or 'Major Issues' (many major issues, clearly biased/deceptive).
            3. 'overall_confidence': (string) A single word indicating your confidence in the overall sentiment (e.g., 'High', 'Medium', 'Low').
            4. 'overall_tone': (string) A single word describing the predominant emotional or attitudinal tone of the content. Choose from: 'Neutral', 'Optimistic', 'Pessimistic', 'Critical', 'Alarmist', 'Sarcastic', 'Humorous', 'Informative'.
            5. 'theme': (string) A single word representing the primary theme or subject of the article (e.g., 'Political', 'Economic', 'Social', 'Environmental', 'Health', 'Technology', 'Education', 'Fashion', 'Sports', 'Entertainment').
            6. 'patterns': A JSON array of objects, where each object represents a detected misinformation pattern.

            Each object in the 'patterns' array MUST have the following keys:
            - 'text_segment': (string) The exact problematic text from the article.
            - 'pattern_type': (string) A clear, concise name for the pattern (e.g., 'Appeal to Emotion', 'False Dichotomy', 'Loaded Language', 'Cherry-Picking Data', 'Ad Hominem', 'Straw Man', 'Slippery Slope', 'Bandwagon Fallacy', 'Generalization', 'Argument from Authority', 'Ad Populum', 'Red Herring').
            - 'explanation': (string) A comprehensive and *detailed* explanation of *how* the pattern is used and *why* it might mislead, considering the overall context and the author's intent. Clearly explain why a statement constitutes a manipulative fallacy rather than a valid critique or strong opinion.
            - 'severity': (string) Assign a severity level: 'Minor' (slight exaggeration, strong opinion, but not fundamentally misleading or deceptive), 'Moderate' (uses a common fallacy, but the core claim might have some basis or the intent to deceive is not overt), or 'Major' (clear distortion of facts, intentional deception, significant logical flaw that severely misleads). Justify the severity.
            - 'confidence_score': (string) A single word indicating your confidence in detecting this specific pattern (e.g., 'High', 'Medium', 'Low').
            - 'trusted_sources_suggestions': (array of strings) Categories of reliable sources that would offer a balanced or factual counter-perspective.

            Ensure the JSON is perfectly formed, with no extra characters, conversational text, or trailing commas outside the single JSON object.
            If no significant manipulative or deceptive patterns are found, the 'patterns' array should be empty (`[]`). If the article is generally fair and balanced, the 'patterns' array might be empty or contain only 'Minor' severity patterns.
            """
        },
        {
            "role": "user",
            "content": f"Analyze the following news article for misinformation patterns:\n\n{article_text}"
        }
    ]

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        data = {
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct", # Updated model
            "messages": messages,
            "max_tokens": 2500, # Increased max tokens for more detailed explanations
            "temperature": 0.3,
            "response_format": {"type": "json_object"}  # Explicitly request JSON object
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=45  # Increased timeout
        )
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        ai_raw_content = response.json()['choices'][0]['message']['content']

        # --- IMPORTANT: Apply the cleaning function here ---
        ai_cleaned_content = clean_json_string(ai_raw_content)

        # Remove markdown code block fences if present (e.g., ```json ... ```)
        if ai_cleaned_content.startswith('```json') and ai_cleaned_content.endswith('```'):
            ai_cleaned_content = ai_cleaned_content[7:-3].strip()
        # Remove extra quotes if AI mistakenly wraps the JSON in string quotes
        elif ai_cleaned_content.startswith('"') and ai_cleaned_content.endswith('"'):
            ai_cleaned_content = ai_cleaned_content[1:-1]
            # Unescape inner quotes if they were escaped by the AI (e.g., \" becomes ")
            ai_cleaned_content = ai_cleaned_content.replace('\\"', '"')

        print(f"AI Raw Response: {ai_raw_content}")  # For debugging
        print(f"AI Cleaned Response (pre-parse): {ai_cleaned_content}")  # For debugging

        return ai_cleaned_content
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Groq API: {e}")
        # Log full response details for debugging 400 errors
        if hasattr(e, 'response') and e.response is not None:
            print(f"API Error Response Status: {e.response.status_code}")
            print(f"API Error Response Body: {e.response.text}")
        return json.dumps({"error": f"Failed to get response from AI: {e}"})
    except KeyError as e:
        print(f"Unexpected API response structure: {response.json()} - {e}")
        return json.dumps({"error": "Unexpected AI response format (KeyError)."})
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from AI response '{ai_cleaned_content[:200]}...': {e}")
        return json.dumps({"error": "AI response was not valid JSON."})
    except Exception as e:
        print(f"An unknown error occurred in ask_ai: {e}")
        return json.dumps({"error": f"An unknown error occurred: {e}"})


# --- Web Scraping Helper (No changes needed) ---
def fetch_article_content(url):
    """Fetches the content of a given URL and extracts the main article text."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        response = requests.get(url, headers=headers, timeout=15)  # Increased timeout
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Try multiple parsers if one fails
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
        except Exception:
            soup = BeautifulSoup(response.text, 'lxml')  # Fallback to lxml if html.parser fails
            # Remember to `pip install lxml`

        # Attempt to find common article content containers
        # Expanded list of common article content selectors and attributes
        article_containers = [
            {'name': 'article'},
            {'name': 'main'},
            {'name': 'div', 'class_': ['article-content', 'post-content', 'story-body', 'entry-content', 'main-content',
                                       'td-post-content', 's--content', 'body__inner', 'g-article__content',
                                       'article-body', 'page-article', 'article__content']},
            {'name': 'div',
             'id': ['content', 'article-body', 'story-page', 'read-more-content', 'main-content', 'single-post-content',
                    'articleText']}
        ]

        extracted_text_parts = []

        # Priority 1: Specific article/main tags or divs with common article classes/ids
        for selector in article_containers:
            if 'class_' in selector:
                elements = soup.find_all(selector['name'], class_=selector['class_'])
            elif 'id' in selector:
                elements = [soup.find(selector['name'], id=cid) for cid in selector['id'] if
                            soup.find(selector['name'], id=cid)]
            else:
                elements = soup.find_all(selector['name'])

            for element in elements:
                # Extract text from paragraphs within these elements
                paragraphs = element.find_all('p')
                if paragraphs:
                    text = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    if text and len(text) > 100:  # Ensure substantial text
                        extracted_text_parts.append(text)
                        # If we find good content in a primary container, consider it sufficient
                        return '\n\n'.join(extracted_text_parts)

        # Priority 2: General fallback for all paragraphs in the body, filtered for length
        if not extracted_text_parts:
            body_paragraphs = soup.find('body').find_all('p') if soup.find('body') else []
            fallback_text = '\n'.join([p.get_text(strip=True) for p in body_paragraphs if
                                       p.get_text(strip=True) and len(
                                           p.get_text(strip=True)) > 50])  # filter short ones
            if fallback_text:
                return fallback_text

        return None  # No significant text found
    except requests.exceptions.RequestException as e:
        return f"Error fetching article: {e}"
    except Exception as e:
        return f"Error parsing article content: {e}"


# --- Helper function to process a single article (reused by index and compare) ---
def process_article(article_url, article_text):
    """
    Fetches content, analyzes it with AI, and formats the results for display.
    Returns a tuple: (analysis_results_dict, error_message_string)
    """
    fetched_content = None
    original_html = None
    error_message = None

    if article_url:
        fetched_content = fetch_article_content(article_url)
        if not fetched_content or "Error" in str(fetched_content):
            error_message = fetched_content if "Error" in str(fetched_content) else "Could not extract meaningful text from the URL. It might be a non-article page, a paywall, or a complex layout."
        else:
            try:
                # Attempt to fetch original HTML for iframe display (can fail due to site blocking)
                response_html_for_iframe = requests.get(article_url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }).text
                original_html = response_html_for_iframe
            except requests.exceptions.RequestException as e:
                print(f"Warning: Could not fetch original HTML for iframe display: {e}")
                original_html = f"<p>Error loading original page for display (might be blocked by website): {e}</p><p>Proceeding with extracted text analysis.</p>"
    elif article_text:
        fetched_content = article_text
        original_html = None # No iframe for direct text input
        if not fetched_content.strip(): # Check if pasted text is just whitespace
            error_message = "Pasted text is empty. Please provide content to analyze."
            fetched_content = None
    else:
        error_message = "Please enter a URL or paste article text."

    analysis_results = None
    if fetched_content and "Error" not in str(fetched_content) and not error_message:
        # Step 2: Ask AI to analyze
        ai_response_json_str = ask_ai(
            prompt="Analyze the following news article for misinformation patterns:",
            article_text=fetched_content
        )

        try:
            # Attempt to parse the AI's response as JSON
            ai_response = json.loads(ai_response_json_str)

            ai_response_list = []  # Initialize as empty list for patterns
            overall_summary = [] # Initialize summary list
            overall_sentiment = "Unknown" # Initialize overall sentiment
            overall_confidence = "N/A" # New: overall confidence
            overall_tone = "N/A" # New: overall tone
            theme = "Unknown" # Initialize theme

            # Check for "patterns" key first, then "data" key, then direct list
            if isinstance(ai_response, dict):
                if "patterns" in ai_response and isinstance(ai_response["patterns"], list):
                    ai_response_list = ai_response["patterns"]
                if "overall_summary" in ai_response and isinstance(ai_response["overall_summary"], list):
                    overall_summary = ai_response["overall_summary"]
                if "overall_sentiment" in ai_response and isinstance(ai_response["overall_sentiment"], str):
                    overall_sentiment = ai_response["overall_sentiment"]
                if "overall_confidence" in ai_response and isinstance(ai_response["overall_confidence"], str): # New: parse overall confidence
                    overall_confidence = ai_response["overall_confidence"]
                if "overall_tone" in ai_response and isinstance(ai_response["overall_tone"], str): # New: parse overall tone
                    overall_tone = ai_response["overall_tone"]
                if "theme" in ai_response and isinstance(ai_response["theme"], str):
                    theme = ai_response["theme"]
                elif "data" in ai_response and isinstance(ai_response["data"], list):
                    # Fallback for older "data" key structure if AI still returns it
                    ai_response_list = ai_response["data"]
            elif isinstance(ai_response, list):
                # If AI returned [...] directly (unlikely with json_object format but for robustness)
                ai_response_list = ai_response
            elif isinstance(ai_response, dict) and "error" in ai_response:
                # Handle explicit error messages from the AI/API
                error_message = f"AI Analysis Error: {ai_response['error']}"
            else:
                # This covers all other unexpected formats
                error_message = f"AI response was not a JSON array of patterns or a dictionary containing them. Unexpected type: {type(ai_response).__name__}. Content: {ai_response_json_str[:200]}..."
                print(
                    f"DEBUG: AI response type unexpected: {type(ai_response)}, Content: {ai_response_json_str}")

            if not error_message:  # Only proceed if no error was set above
                analysis_results_display = []
                highlighted_text = fetched_content  # Start with original fetched content

                # Use ai_response_list for sorting and iteration
                # Sort patterns by the length of the text_segment in descending order
                sorted_patterns = sorted(ai_response_list, key=lambda x: len(x.get('text_segment', '')),
                                         reverse=True)

                for i, pattern in enumerate(sorted_patterns):
                    segment = pattern.get('text_segment')
                    pattern_type = pattern.get('pattern_type', 'Unknown Pattern')
                    explanation = pattern.get('explanation', 'No explanation provided.')
                    severity = pattern.get('severity', 'Unknown') # Extract severity
                    confidence_score = pattern.get('confidence_score', 'N/A') # New: get confidence score for pattern
                    ai_suggestions = pattern.get('trusted_sources_suggestions', [])

                    actual_sources = []
                    for suggestion_category in ai_suggestions:
                        # Try direct match first (case-insensitive)
                        matched = False
                        for key, links in TRUSTED_SOURCE_LINKS.items():
                            if suggestion_category.lower() == key.lower():
                                actual_sources.extend(links)
                                matched = True
                                break
                        # If no direct match, try partial/fuzzy match (contains logic)
                        if not matched:
                            for key, links in TRUSTED_SOURCE_LINKS.items():
                                if suggestion_category.lower() in key.lower() or key.lower() in suggestion_category.lower():
                                    actual_sources.extend(links)

                    # Ensure unique sources (by URL)
                    actual_sources_unique = list({s['url']: s for s in actual_sources}.values())

                    analysis_results_display.append({
                        'id': f'pattern-{i}',
                        'pattern_type': pattern_type,
                        'explanation': explanation,
                        'severity': severity, # Add severity to results
                        'confidence_score': confidence_score, # New: add to display list
                        'text_segment': segment,
                        'trusted_sources_suggestions': actual_sources_unique  # Now contains actual links
                    })

                    # Highlight the text segment if found
                    if segment and segment in highlighted_text:
                        # Build sources HTML for the tooltip
                        tooltip_sources_html = ""
                        if actual_sources_unique:
                            tooltip_sources_html = '<br>'.join(
                                [f"<a href='{s['url']}' target='_blank'>{s['name']}</a>" for s in
                                 actual_sources_unique])
                        else:
                            tooltip_sources_html = 'No specific links available.'

                        # Replace the segment with the highlighted span including the tooltip
                        # Using a unique ID for each highlight to link to the pattern details
                        highlighted_text = highlighted_text.replace(
                            segment,
                            f"<span class='highlight highlight-{severity.lower().replace(' ', '-')}' data-pattern-id='pattern-{i}'>{segment}"
                            f"<div class='tooltip-content'>"
                            f"<b>{pattern_type}:</b> {explanation}<br>"
                            f"<b>Severity:</b> {severity}<br>" # Display severity in tooltip
                            f"<b>Confidence:</b> {confidence_score}<br><br>" # New: Display confidence in tooltip
                            f"<b>Sources:</b> {tooltip_sources_html}"
                            f"</div></span>"
                        )

                analysis_results = {
                    'original_url': article_url if article_url else "Direct Text Input", # Indicate text input
                    'extracted_text': fetched_content,
                    'highlighted_text': highlighted_text,  # For displaying extracted text with highlights
                    'patterns': analysis_results_display,
                    'overall_summary': overall_summary, # Add overall summary to results
                    'overall_sentiment': overall_sentiment, # Add overall sentiment
                    'overall_confidence': overall_confidence, # New: Add overall confidence to results
                    'overall_tone': overall_tone, # New: Add overall tone to results
                    'theme': theme, # Add theme
                    'original_html_content': original_html  # For displaying the full page
                }

        except json.JSONDecodeError as e:
            error_message = f"AI returned malformed JSON: {e}. Raw AI response: {ai_response_json_str[:500]}..."
            print(f"DEBUG: JSON Decode Error, raw AI response: {ai_response_json_str}")
        except Exception as e:
            error_message = f"An unexpected error occurred during AI response processing: {e}"

    return analysis_results, error_message


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_article():
    """Handles single article analysis via AJAX."""
    article_url_input = request.form.get('article_url', '')
    article_text_input = request.form.get('article_text', '')

    analysis_results, error_message = process_article(article_url_input, article_text_input)

    if error_message:
        return jsonify({"error": error_message}), 400
    if analysis_results:
        return jsonify(analysis_results)
    return jsonify({"error": "No analysis results generated."}), 500


@app.route('/compare', methods=['POST'])
def compare_articles():
    """Handles comparison of two articles via AJAX."""
    article1_url = request.form.get('article1_url', '')
    article1_text = request.form.get('article1_text', '')
    article2_url = request.form.get('article2_url', '')
    article2_text = request.form.get('article2_text', '')

    results1, error1 = process_article(article1_url, article1_text)
    results2, error2 = process_article(article2_url, article2_text)

    response_data = {
        "article1": results1,
        "article2": results2,
        "error1": error1,
        "error2": error2
    }
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
