import os
import sys
import asyncio
import platform

# --- Import Checks ---
try:
    import streamlit as st
    from streamlit.web import cli as stcli
    from browser_use import Agent
    from langchain_openai import ChatOpenAI
    from browser_use.browser.browser import Browser, BrowserConfig
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("Run: pip install streamlit browser-use playwright langchain-openai langchain-anthropic")
    sys.exit(1)

# --- Core Logic ---

async def run_agent_with_debug(query, model_choice, api_key, headless, status_box):
    # 1. Setup OpenRouter Headers
    headers = {"HTTP-Referer": "http://localhost:8501", "X-Title": "MemeGen Debug"}

    # 2. Configure Model
    if model_choice == "Claude":
        model = "meta-llama/llama-3.3-70b-instruct:free"
        temp = 0.7
    elif model_choice == "Deepseek":
        model = "nex-agi/deepseek-v3.1-nex-n1:free"
        temp = 0.3
    else: # OpenAI
        model = "openai/gpt-oss-120b:free"
        temp = 0.1

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model=model,
        api_key=api_key,
        default_headers=headers,
        temperature=temp,
    )

    # 3. Task
    task = (
        f"Generate a meme about: '{query}'.\n"
        "STRICT STEPS:\n"
        "1. GO TO 'https://imgflip.com/memetemplates'.\n"
        "2. TYPE a simple keyword related to '{query}' into the search box.\n"
        "3. CLICK on the first matching template image.\n"
        "4. WAIT for the 'Caption this Meme' button or text boxes to load.\n"
        "5. CLICK inside the Top Text box and TYPE a setup.\n"
        "6. CLICK inside the Bottom Text box and TYPE a punchline.\n"
        "7. CLICK 'Generate Meme'.\n"
        "8. EXTRACT and RETURN the image URL starting with 'https://i.imgflip.com'.\n"
    )

    # 4. Browser Setup
    # CRITICAL: Headless must be True on servers/codespaces
    browser = Browser(config=BrowserConfig(headless=headless))
    
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        use_vision=False, # Disable vision to prevent Free Model hangs
        max_failures=3,
        max_actions_per_step=1
    )

    status_box.info(f"🚀 Launching Browser (Headless: {headless})...")
    
    final_result = None
    errors = []

    try:
        history = await agent.run()
        final_result = history.final_result()
    except Exception as e:
        errors.append(str(e))
    finally:
        await browser.close()

    return final_result, errors

# --- UI ---

def main():
    st.set_page_config(page_title="AI Meme Debugger", page_icon="🐞")

    st.title("🐞 AI Meme Generator (Fixed)")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        # Added key="api_key_input" to fix duplicate ID error
        api_key = st.text_input("OpenRouter Key", type="password", key="api_key_input")
        
        # Added key="model_select"
        model = st.selectbox("Model", ["Deepseek", "Claude", "OpenAI"], key="model_select")
        
        # Headless defaults to True on Linux (Codespaces)
        is_cloud = platform.system() == "Linux"
        headless = st.checkbox("Headless Mode", value=is_cloud, key="headless_check")
        
    # Main Input
    # Added key="meme_query_input" to fix duplicate ID error
    query = st.text_input("Meme Prompt", "Programmers working on friday", key="meme_query_input")

    if st.button("Run Agent", key="run_button"):
        if not api_key:
            st.error("Need API Key")
            st.stop()

        status = st.empty()
        log_expander = st.expander("📝 Live Logs", expanded=True)
        
        with st.spinner("Agent is working..."):
            try:
                # Capture stdout to show logs in UI
                import io
                from contextlib import redirect_stdout
                
                f = io.StringIO()
                with redirect_stdout(f):
                    result, errors = asyncio.run(
                        run_agent_with_debug(query, model, api_key, headless, status)
                    )
                logs = f.getvalue()
                log_expander.code(logs)

                if errors:
                    st.error("💥 The Agent crashed!")
                    st.json(errors)
                
                if result:
                    st.success("Agent Finished!")
                    st.markdown(f"### Result:\n{result}")
                    
                    import re
                    url_match = re.search(r'(https://i\.imgflip\.com/\w+\.jpg)', result)
                    if url_match:
                        st.image(url_match.group(1))
                    else:
                        st.warning("No image link found in result.")
                else:
                    st.warning("Agent finished but returned no result.")

            except Exception as e:
                st.error(f"Critical Error: {e}")

if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
