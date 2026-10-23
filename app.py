"""
Study Planner Assistant - a small multi-agent system.

Agents:
  1. PlannerAgent   - builds a weekly study schedule
  2. ResourceAgent  - suggests topics and study tips per subject
  3. QuizAgent      - generates practice questions on a chosen topic
Coordinator:
  Passes the user's input to each agent in turn, logs every step,
  and returns the combined result to the UI.

LLM: Groq free API (llama-3.3-70b-versatile)
UI : Streamlit
"""

import os
import time
from datetime import datetime

import streamlit as st
from groq import Groq

MODEL = "openai/gpt-oss-120b"


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------
class Agent:
    """A single agent: a name, a role (system prompt) and a run() method."""

    def __init__(self, name: str, role: str, client: Groq):
        self.name = name
        self.role = role
        self.client = client

    def run(self, task: str) -> str:
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.role},
                {"role": "user", "content": task},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content


class PlannerAgent(Agent):
    def __init__(self, client):
        super().__init__(
            "Planner Agent",
            "You are a study planning expert. Create a clear day-by-day weekly "
            "study schedule as a markdown table. Give harder or nearer-exam "
            "subjects more time. Include short breaks. Keep it concise.",
            client,
        )


class ResourceAgent(Agent):
    def __init__(self, client):
        super().__init__(
            "Resource Agent",
            "You are an academic mentor. For each subject, list 4-5 key topics "
            "to prioritise and 2-3 practical study tips. Use short bullet points.",
            client,
        )


class QuizAgent(Agent):
    def __init__(self, client):
        super().__init__(
            "Quiz Agent",
            "You are an examiner. Create exactly 5 multiple-choice questions on "
            "the given topic, each with 4 options, then give an answer key at "
            "the end. Mix easy and medium difficulty.",
            client,
        )


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------
class Coordinator:
    """Routes the user's request to each agent and records an activity log."""

    def __init__(self, client: Groq):
        self.planner = PlannerAgent(client)
        self.resource = ResourceAgent(client)
        self.quiz = QuizAgent(client)
        self.log = []

    def _call(self, agent: Agent, task: str) -> str:
        start = time.time()
        self._log(f"Coordinator -> {agent.name}: task assigned")
        try:
            output = agent.run(task)
            status = "completed"
        except Exception as exc:  # keep the app alive if one agent fails
            output = f"Error from {agent.name}: {exc}"
            status = "failed"
        elapsed = time.time() - start
        self._log(f"{agent.name}: {status} in {elapsed:.1f}s")
        return output

    def _log(self, message: str):
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def run(self, subjects: str, exam_info: str, hours: int, quiz_topic: str):
        self._log("Coordinator: request received")

        plan = self._call(
            self.planner,
            f"Subjects: {subjects}\nExam dates: {exam_info}\n"
            f"Study hours available per day: {hours}\n"
            "Create the weekly schedule.",
        )
        resources = self._call(
            self.resource,
            f"Subjects: {subjects}\nSuggest priority topics and study tips.",
        )
        quiz = self._call(
            self.quiz,
            f"Topic: {quiz_topic}\nGenerate the quiz.",
        )

        self._log("Coordinator: all agents finished, results combined")
        return plan, resources, quiz, self.log


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


def main():
    st.set_page_config(page_title="Study Planner Assistant", page_icon="📚")
    st.title("📚 Study Planner Assistant")
    st.caption("A multi-agent system: Planner + Resource + Quiz agents, "
               "managed by a Coordinator.")

    api_key = get_api_key()
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it in Streamlit secrets or as an "
                 "environment variable.")
        st.stop()

    with st.form("inputs"):
        subjects = st.text_input(
            "Subjects (comma separated)", "Data Structures, DBMS, Maths"
        )
        exam_info = st.text_input(
            "Exam dates", "DS - 5 Nov, DBMS - 8 Nov, Maths - 12 Nov"
        )
        hours = st.slider("Study hours per day", 1, 12, 4)
        quiz_topic = st.text_input("Quiz topic", "Binary Search Trees")
        submitted = st.form_submit_button("Run all agents")

    if submitted:
        coordinator = Coordinator(Groq(api_key=api_key))
        with st.spinner("Agents are working..."):
            plan, resources, quiz, log = coordinator.run(
                subjects, exam_info, hours, quiz_topic
            )

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🗓️ Planner Agent", "📖 Resource Agent", "📝 Quiz Agent",
             "🧾 Agent Activity Log"]
        )
        with tab1:
            st.markdown(plan)
        with tab2:
            st.markdown(resources)
        with tab3:
            st.markdown(quiz)
        with tab4:
            st.code("\n".join(log), language="text")


if __name__ == "__main__":
    main()
