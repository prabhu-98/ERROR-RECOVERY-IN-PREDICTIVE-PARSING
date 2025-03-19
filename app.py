import streamlit as st
import google.generativeai as genai

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDQRPWOtlzJZFWwGiU7j3fg9_gUdsPNFsU"
genai.configure(api_key=GEMINI_API_KEY)

# Streamlit UI
st.title("LL(1) Predictive Parser with AI-based Error Recovery")

# User Input for Grammar
st.subheader("Enter LL(1) Grammar")
grammar_input = st.text_area(
    "Define grammar (format: NonTerminal -> Production1 | Production2)",
    "E -> T E'\nE' -> + T E' | ε\nT -> F T'\nT' -> * F T' | ε\nF -> ( E ) | id"
)

# User Input for Parsing String
st.subheader("Enter Input String")
user_input = st.text_input("Enter a space-separated sequence of tokens", "id + id * id")

# Parse grammar input into dictionary format
def parse_grammar(grammar_text):
    grammar = {}
    for line in grammar_text.split("\n"):
        line = line.strip()
        if "->" in line:
            nt, rhs = line.split("->")
            nt = nt.strip()
            productions = [prod.strip().split() for prod in rhs.split("|")]
            grammar[nt] = productions
    return grammar

# LL(1) Grammar Parsing
def parse(tokens, grammar):
    stack = ['$', 'E']
    index = 0
    steps = []

    while stack:
        top = stack.pop()
        steps.append(f"Stack: {stack}, Current Token: {tokens[index] if index < len(tokens) else 'EOF'}")

        if top == 'ε':  # Ignore epsilon
            continue
        elif index < len(tokens) and top == tokens[index]:  # Match terminal
            steps.append(f"Matched terminal: {top}")
            index += 1
        elif top in grammar:  # Non-terminal: apply rule
            production = get_production(top, tokens[index], grammar)
            if production:
                steps.append(f"Expanding {top} → {' '.join(production)}")
                stack.extend(reversed(production))
            else:
                error_context = f"Error: Unexpected token '{tokens[index]}' when expecting '{top}'. Suggest a correction."
                correction = ai_error_recovery(error_context)

                if correction and (correction in grammar or correction in ['id', '+', '*', '(', ')']):
                    steps.append(f"🔴 Error at '{tokens[index]}' → 🔄 Suggested Replacement: '{correction}'")
                    tokens[index] = correction  # Correct token
                else:
                    steps.append(f"❌ Parsing failed. No valid correction found for '{tokens[index]}'.")
                    return False, steps, None
        else:
            steps.append(f"❌ Unexpected token '{tokens[index]}'. Parsing failed.")
            return False, steps, tokens[index]

    return True, steps, None

# Get production for a non-terminal
def get_production(non_terminal, token, grammar):
    for production in grammar.get(non_terminal, []):
        if production[0] == token or production[0] == 'ε':
            return production
    return None

# AI-based error recovery
def ai_error_recovery(context):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([context], generation_config={"temperature": 0.3, "max_output_tokens": 5})
    suggestion = response.text.strip() if response and hasattr(response, 'text') else None
    return suggestion.split()[0] if suggestion else None

# Processing on button click
if st.button("Parse Input"):
    grammar = parse_grammar(grammar_input)
    tokens = user_input.split()
    success, steps, error_token = parse(tokens, grammar)

    st.subheader("Parsing Results")
    if success:
        st.success("✅ Parsing Successful!")
    else:
        st.error(f"❌ Parsing Failed! Error at token '{error_token}'.")

    st.subheader("Processing Steps")
    for step in steps:
        st.text(step)
