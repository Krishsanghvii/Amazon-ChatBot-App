import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import load_dotenv

import pandas as pd
from model import PredictReview  # Using your enhanced model

# Global variables to store model and metrics
model_metrics = None

#Get Sentiment with Model Evaluation
def get_sentiment(text, show_metrics=False):
    global model_metrics
    #importing dataset
    data = pd.read_csv("output.csv")
    data['label_num'] = pd.get_dummies(data['label'],drop_first=True)

    review_predictor = PredictReview()
    model, converter, metrics = review_predictor.base(data)
    
    # Store metrics globally
    model_metrics = metrics
    
    answer = review_predictor.test_sample(text, converter, model)
    
    if show_metrics:
        return answer, metrics
    return answer

# load the Environment Variables. 
load_dotenv()
st.set_page_config(page_title="Amazon Product App")

# Sidebar contents
with st.sidebar:
    st.title('Amazon Product Related Queries App 🤗💬')
    st.markdown('''
    ## About
    This app is an Review Sentiment Analysis and a LLM-powered chatbot for Amazon Product related queries:
    ''')
    menu = ['Amazon Review Sentiment Analysis','Product Queries BOT', 'Model Performance']
    choice = st.sidebar.selectbox("Select an option", menu)
    add_vertical_space(10)
    st.write('Made by [Krish Sanghvi](https://github.com/Krishsanghvii)')

st.header("Your Amazon Assistant 💬")
st.divider()

def main():
    global model_metrics

    if choice == 'Amazon Review Sentiment Analysis':
        st.subheader("Amazon Review Sentiment Analysis")
        with st.form(key='my_form'):
            raw_text = st.text_area("Enter the amazon review here:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.info("Sentiment:")
            answer, metrics = get_sentiment(raw_text, show_metrics=True)
            st.write(answer)
            
            # Show model performance metrics
            if metrics:
                with st.expander("Model Performance Metrics"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.3f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.3f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")

    elif choice == 'Model Performance':
        st.subheader("Model Performance Dashboard")
        
        if model_metrics is None:
            # Train model to get metrics
            data = pd.read_csv("output.csv")
            data['label_num'] = pd.get_dummies(data['label'],drop_first=True)
            review_predictor = PredictReview()
            _, _, model_metrics = review_predictor.base(data)
        
        if model_metrics:
            st.write("### Classification Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{model_metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{model_metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{model_metrics['f1_score']:.4f}")
            
            st.write("### Confusion Matrix")
            st.write(model_metrics['confusion_matrix'])
            
            # Additional insights
            st.write("### Model Insights")
            st.write(f"- **True Negatives:** {model_metrics['confusion_matrix'][0][0]}")
            st.write(f"- **False Positives:** {model_metrics['confusion_matrix'][0][1]}")
            st.write(f"- **False Negatives:** {model_metrics['confusion_matrix'][1][0]}")
            st.write(f"- **True Positives:** {model_metrics['confusion_matrix'][1][1]}")

    elif choice == 'Product Queries BOT':
        st.subheader("Product Queries BOT")    
        # Generate empty lists for generated and user.
        ## Assistant Response
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hi, please ask your queries related to amazon products?"]

        ## user question
        if 'user' not in st.session_state:
            st.session_state['user'] = ['Hi!']

        # Layout of input/response containers
        response_container = st.container()
        colored_header(label='', description='', color_name='blue-30')
        input_container = st.container()

        # get user input
        def get_text():
            input_text = st.text_input("You: ", "", key="input")
            return input_text

        ## Applying the user input box
        with input_container:
            user_input = get_text()

        def chain_setup():
            template = """Your are amazon product related query bot so answer only product related questions, if any other questions asked then don't answer: <|prompter|>{question}<|endoftext|>
            <|assistant|>"""
            
            prompt = PromptTemplate(template=template, input_variables=["question"])
            llm=HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", model_kwargs={"max_new_tokens":1200})
            llm_chain=LLMChain(llm=llm, prompt=prompt)
            return llm_chain

        # generate response
        def generate_response(question, llm_chain):
            response = llm_chain.run(question)
            return response

        ## load LLM
        llm_chain = chain_setup()

        # main loop
        with response_container:
            if user_input:
                response = generate_response(user_input, llm_chain)
                st.session_state.user.append(user_input)
                st.session_state.generated.append(response)
                
            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['user'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))

if __name__ == '__main__':
    main()
