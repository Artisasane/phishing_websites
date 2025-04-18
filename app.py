import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse, urlencode # Added urlencode
import tldextract # You might need to install: pip install tldextract

# --- Feature Extraction (Simplified & Incomplete) ---
# WARNING: This extracts only a fraction of the original 87 features.
# It CANNOT correctly generate the input needed for the PCA-based model.

def count_special_chars(url):
    """Counts various special characters in a URL."""
    counts = {
        'nb_dots': url.count('.'),
        'nb_hyphens': url.count('-'),
        'nb_at': url.count('@'),
        'nb_qm': url.count('?'),
        'nb_and': url.count('&'),
        'nb_or': url.count('|'), # Note: '|' is uncommon in valid URLs
        'nb_eq': url.count('='),
        'nb_underscore': url.count('_'),
        'nb_tilde': url.count('~'),
        'nb_percent': url.count('%'),
        'nb_slash': url.count('/'),
        'nb_star': url.count('*'), # Note: '*' is uncommon in valid URLs
        'nb_colon': url.count(':'),
        'nb_comma': url.count(','),
        'nb_semicolumn': url.count(';'),
        'nb_dollar': url.count('$'),
        'nb_space': url.count(' ') # Spaces should be encoded, but check anyway
    }
    return counts

def get_url_features(url):
    """Extracts features derivable SOLELY from the URL string."""
    features = {}
    try:
        # Basic Lengths
        features['length_url'] = len(url)
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc
        path = parsed_url.path
        features['length_hostname'] = len(hostname)

        # Special Character Counts
        features.update(count_special_chars(url))

        # Hostname-based features
        features['nb_www'] = hostname.lower().count('www')
        # Simple check for .com in hostname (might be inaccurate for subdomains)
        features['nb_com'] = 1 if '.com' in hostname.lower() else 0

        # Protocol
        features['https_token'] = 1 if parsed_url.scheme == 'https' else 0

        # Digits Ratio
        digits_url = sum(c.isdigit() for c in url)
        digits_host = sum(c.isdigit() for c in hostname)
        features['ratio_digits_url'] = digits_url / features['length_url'] if features['length_url'] > 0 else 0
        features['ratio_digits_host'] = digits_host / features['length_hostname'] if features['length_hostname'] > 0 else 0

        # TLD related (simplified)
        try:
            tld_info = tldextract.extract(url)
            features['tld_in_subdomain'] = 1 if tld_info.subdomain and tld_info.tld in tld_info.subdomain else 0
            features['tld_in_path'] = 1 if path and tld_info.tld in path else 0
            features['nb_subdomains'] = len(tld_info.subdomain.split('.')) if tld_info.subdomain else 0
        except Exception: # Handle potential tldextract errors
             features['tld_in_subdomain'] = 0
             features['tld_in_path'] = 0
             features['nb_subdomains'] = 0


        # Other simple checks
        features['nb_dslash'] = path.count('//')
        features['http_in_path'] = 1 if 'http:' in path.lower() or 'https:' in path.lower() else 0
        features['prefix_suffix'] = 1 if '-' in hostname else 0 # Simple check for hyphen in host

        # --- Add more URL-based features here if needed ---
        # Example: port, punycode, specific path extensions, etc.
        # features['punycode'] = 1 if 'xn--' in hostname.lower() else 0
        # features['port'] = parsed_url.port if parsed_url.port else 0 # Often needs specific handling

    except Exception as e:
        st.error(f"Error parsing URL or extracting features: {e}")
        # Return a dictionary with default values or NaNs if parsing fails
        # For simplicity, returning zeros. A real app needs robust error handling.
        feature_names = ['length_url', 'length_hostname', 'nb_dots', 'nb_hyphens', 'nb_at',
                         'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore', 'nb_tilde',
                         'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma',
                         'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 'nb_com',
                         'https_token', 'ratio_digits_url', 'ratio_digits_host',
                         'tld_in_subdomain', 'tld_in_path', 'nb_subdomains',
                         'nb_dslash', 'http_in_path', 'prefix_suffix'] # Add others if implemented
        features = {name: 0 for name in feature_names}


    # Ensure all expected columns (that we *can* calculate) are present, even if 0
    # This list should match the features calculated above
    expected_url_features = list(features.keys()) # Dynamically get keys from calculated features
    for f in expected_url_features:
        if f not in features:
            features[f] = 0 # Default to 0 if calculation failed for some reason

    return features

# --- Load Model (and hypothetical preprocessors) ---
# IMPORTANT: This assumes the model, scaler, and PCA were saved previously.
# If scaler.pkl and pca.pkl don't exist, this will fail.
model_path = 'best_rf_model.pkl'
# scaler_path = 'scaler.pkl' # Hypothetical path
# pca_path = 'pca.pkl'       # Hypothetical path

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}")
    st.stop() # Stop execution if model isn't found
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# try:
#     with open(scaler_path, 'rb') as f:
#         scaler = pickle.load(f)
# except FileNotFoundError:
#     st.warning(f"Scaler file not found at {scaler_path}. Cannot perform scaling.")
#     scaler = None # Set scaler to None if not found
# except Exception as e:
#     st.error(f"Error loading the scaler: {e}")
#     scaler = None

# try:
#     with open(pca_path, 'rb') as f:
#         pca = pickle.load(f)
# except FileNotFoundError:
#     st.warning(f"PCA file not found at {pca_path}. Cannot perform PCA.")
#     pca = None # Set PCA to None if not found
# except Exception as e:
#     st.error(f"Error loading PCA: {e}")
#     pca = None


# --- Streamlit App Interface ---
st.set_page_config(page_title="Phishing Detector", layout="wide")
st.title("🎣 Phishing URL Detector")
st.markdown("""
Enter a URL below to check if it's likely a phishing site.

**Disclaimer:** This tool uses a machine learning model trained on specific features.
*   The original model relied on features requiring external data (like web traffic, WHOIS) and page content analysis, which **cannot be performed** by simply entering a URL here.
*   Therefore, the prediction accuracy might be **significantly lower** than the original model's performance.
*   This app serves primarily as a **demonstration** of deploying a model with Streamlit. **Do not rely solely on this tool for security decisions.**
""")

url_input = st.text_input("Enter the URL to check:", placeholder="e.g., https://www.google.com")

if st.button("Check URL"):
    if url_input:
        st.write(f"Analyzing URL: `{url_input}`")

        # 1. Extract URL-based features (Incomplete Set)
        raw_features = get_url_features(url_input)
        feature_df = pd.DataFrame([raw_features])

        st.write("Extracted URL Features (Subset):")
        st.dataframe(feature_df.head()) # Display extracted raw features

        # --- !! CRITICAL FLAW !! ---
        # The following steps are fundamentally incorrect because:
        # a) We don't have all 87 original features.
        # b) We don't have the *actual* saved scaler and PCA objects.
        # We simulate the *expected input shape* for the model (2 PCs).
        # THIS WILL NOT PRODUCE VALID PREDICTIONS.

        # Placeholder for Preprocessing (Scaling + PCA) Simulation
        # In a real scenario with saved scaler/pca and *all* features:
        # Check if feature_df columns match scaler's expected features
        # scaled_features = scaler.transform(feature_df)
        # pca_features = pca.transform(scaled_features)
        # input_data = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])

        # --- !!! SIMULATED INPUT FOR DEMO !!! ---
        # Creating dummy PC data because we can't generate the real ones.
        # The loaded model expects 2 input features (PC1, PC2).
        st.warning("Simulating PCA input due to missing features/preprocessors. Prediction is illustrative only.")
        # Create dummy data with the correct shape (1 row, 2 columns)
        # Replace this with actual pca_features if preprocessing was possible
        dummy_pca_input = np.array([[0.0, 0.0]]) # Example dummy values
        input_data = pd.DataFrame(dummy_pca_input, columns=['PC1', 'PC2']) # Match expected input names if known

        # 3. Make Prediction using the loaded model
        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            # 4. Display Result
            st.subheader("Prediction Result:")
            is_phishing = prediction[0] # Model output is likely 0 (legit) or 1 (phishing) based on LabelEncoder

            if is_phishing == 1:
                st.error(f"🚨 **Phishing Alert!**")
                st.write(f"This URL is classified as **Phishing** with a confidence of **{prediction_proba[0][1]:.2f}**.")
            else:
                st.success(f"✅ **Likely Legitimate**")
                st.write(f"This URL is classified as **Legitimate** with a confidence of **{prediction_proba[0][0]:.2f}**.")

            st.write("Probability [Legitimate, Phishing]:", prediction_proba[0])

        except Exception as e:
             st.error(f"Error during prediction: {e}")
             st.error("Could not make a prediction. This might be due to the mismatch between extracted features and model expectations.")

    else:
        st.warning("Please enter a URL.")

st.markdown("---")
st.markdown("Developed as a demonstration.")