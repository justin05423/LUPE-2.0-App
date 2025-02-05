import streamlit as st
from utils.import_utils import *
from utils.visuals_utils import *
from stqdm import stqdm


def app():
    st.title("Analysis Page")
    st.markdown("Welcome to the analysis page!")

    # Existing code starts here
    def prompt_setup():
        st.file_uploader('upload csv file', accept_multiple_files=True)

    files = [st.session_state[f'fnames_condition_{n + 1}'] for n in range(st.session_state['num_condition'])]
    if not st.session_state.extracted:
        if not all(files):
            st.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                        f"font-family:Avenir; font-weight:normal'>Please Upload Files...</h1> "
                        , unsafe_allow_html=True)
        elif all(files):
            # [All your logic remains unchanged]
            ...

    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"LUPE is developed by Alexander Hsu and Justin James</h1> "
                    , unsafe_allow_html=True)