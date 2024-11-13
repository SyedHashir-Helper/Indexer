import streamlit as st
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('averaged_perceptron_tagger') 
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize global variables in session state
if 'documents' not in st.session_state:
    st.session_state.documents = {}  # Dictionary to store document titles and content
if 'index' not in st.session_state:
    st.session_state.index = defaultdict(list)  # Hash table to store words/synonyms and associated document IDs
if 'stop_words' not in st.session_state:
    st.session_state.stop_words = set(stopwords.words('english'))
if 'lemmatizer' not in st.session_state:
    st.session_state.lemmatizer = WordNetLemmatizer()
if 'word_counter' not in st.session_state:
    st.session_state.word_counter = Counter()  # Counter to store word frequencies across documents

# Set a threshold for term frequency (e.g., words appearing more than once)
TERM_FREQUENCY_THRESHOLD = 1

import nltk
nltk.download('averaged_perceptron_tagger')  # Ensure POS tagger is downloaded

def preprocess(text):
    """Preprocess the text by keeping only nouns, removing stop words, punctuation, and lemmatizing."""
    # Tokenize and remove punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Part-of-speech tagging
    pos_tags = nltk.pos_tag(words)
    
    # Filter and lemmatize only nouns
    filtered_words = [
        st.session_state.lemmatizer.lemmatize(word)
        for word, pos in pos_tags
        if pos in ('NN', 'NNS', 'NNP', 'NNPS') and word not in st.session_state.stop_words
    ]
    return filtered_words


def get_synonyms(word):
    """Retrieve synonyms for a word using WordNet and convert them to lowercase."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()  # Convert synonym to lowercase
            synonyms.add(synonym)
    return synonyms

def build_index(doc_id, content):
    """Build an index with hash tables for the new document, using only relevant words."""
    words = preprocess(content)
    word_counts = Counter(words)  # Calculate word frequency
    st.session_state.word_counter.update(word_counts)  # Update global word frequency counter
    
    # Only include words that meet or exceed the threshold frequency
    relevant_words = {word for word, count in word_counts.items() if count > TERM_FREQUENCY_THRESHOLD}
    
    for word in relevant_words:
        # Insert document ID for the word itself
        if doc_id not in st.session_state.index[word]:
            st.session_state.index[word].append(doc_id)
        # Insert document ID for each synonym
        for synonym in get_synonyms(word):
            if doc_id not in st.session_state.index[synonym]:
                st.session_state.index[synonym].append(doc_id)

def search_by_title(query):
    """Search for a document by title (partial match)."""
    query_words = set(query.lower().split())
    results = []
    for doc_id, doc_data in st.session_state.documents.items():
        title, _ = doc_data
        title_words = set(title.lower().split())
        if query_words.intersection(title_words):
            results.append(f"Found in document {doc_id} titled '{title}'")
    return results if results else ["No document with that title found."]

def search_by_content(query):
    """Search for documents containing all specific words or their synonyms in content."""
    query_words = preprocess(query)
    matching_documents = None  # Start with no matches to perform intersection

    for word in query_words:
        # Get the documents that contain the word or any of its synonyms
        word_matches = set(st.session_state.index.get(word, []))
        for synonym in get_synonyms(word):
            word_matches.update(st.session_state.index.get(synonym, []))

        # If no documents contain the word or any of its synonyms, return early
        if not word_matches:
            return "No documents match the search content."

        # If it's the first word, initialize the matching_documents set with its matches
        if matching_documents is None:
            matching_documents = word_matches
        else:
            # Perform intersection to keep only documents containing all query words/synonyms
            matching_documents &= word_matches

    # If we found matching documents, list them; otherwise, no results
    if matching_documents:
        doc_titles = [f"{doc_id} titled '{st.session_state.documents[doc_id][0]}'" for doc_id in matching_documents]
        return f"Documents found: {', '.join(doc_titles)}"
    return "No documents match the search content."


def plot_most_common_words():
    """Plot the five most common words in all documents."""
    if not st.session_state.word_counter:
        st.write("No words to display yet. Please upload and index documents first.")
        return
    
    # Get the five most common words and their frequencies
    common_words = st.session_state.word_counter.most_common(5)
    words, counts = zip(*common_words)
    
    # Plot using matplotlib
    fig, ax = plt.subplots()
    ax.bar(words, counts, color='skyblue')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 5 Most Common Words in Documents')
    
    # Display the plot in Streamlit
    st.pyplot(fig)

# Streamlit UI
st.title("Simple Document Search Engine")

# Sidebar - Document List
st.sidebar.title("Available Documents")
for doc_id, doc_data in st.session_state.documents.items():
    st.sidebar.write(f"{doc_id} - {doc_data[0]}")

# File upload and confirmation
uploaded_files = st.file_uploader("Upload text files", type="txt", accept_multiple_files=True)
if uploaded_files:
    confirm_upload = st.button("Confirm Upload")
    if confirm_upload:
        for uploaded_file in uploaded_files:
            # Read and index each document after confirmation
            lines = uploaded_file.read().decode("utf-8").splitlines()
            
            # Ignore the first line, then pick the title from the second line after "Title:"
            if len(lines) < 2:
                st.error(f"File {uploaded_file.name} does not have the expected format.")
                continue
            title = lines[1].replace("Title:", "").strip()  # Remove "Title:" and strip any extra whitespace
            content = "\n".join(lines[2:])  # The rest of the lines form the content

            # Create document ID and store document title and content
            doc_id = f"D{len(st.session_state.documents) + 1}"
            st.session_state.documents[doc_id] = (title, content)
            build_index(doc_id, content)
            st.sidebar.write(f"{doc_id} - {title}")
        
        st.success(f"{len(uploaded_files)} documents indexed.")
        

# Search options
st.write("## Search Documents")
query = st.text_input("Enter your search query")
search_option = st.selectbox("Search by", ["Content", "Title"], index=0)  # Non-editable dropdown
search_button = st.button("Search")

# Search action
if search_button:
    if search_option == "Content":
        result = search_by_content(query)
    else:
        result = search_by_title(query)
    st.write("### Search Results")
    st.write(result)

plot_most_common_words()  # Display the most common words after indexing