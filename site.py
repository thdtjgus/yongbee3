import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 엑셀 파일 경로 설정 (실제 파일 경로로 변경하세요)
EXCEL_FILE_PATH = 'asdf.xlsx'

# 엑셀 파일 로드 및 데이터 전처리
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df['Q'] = df['Q'].fillna('')
    df['A'] = df['A'].fillna('')
    return df

df = load_data(EXCEL_FILE_PATH)

# TF-IDF 벡터화 도구 학습
@st.cache_resource
def train_vectorizer(data):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(data['Q'].tolist())
    return vectorizer, vectors

question_vectorizer, question_vector = train_vectorizer(df)

# 유사 질문 찾기 함수
def get_most_similar_question(user_question, threshold):
    new_sen_vector = question_vectorizer.transform([user_question])
    simil_score = cosine_similarity(new_sen_vector, question_vector)
    if simil_score.max() < threshold:
        return None, "유사한 질문을 찾을 수 없습니다."
    else:
        max_index = simil_score.argmax()
        most_similar_question = df['Q'].tolist()[max_index]
        most_similar_answer = df['A'].tolist()[max_index]
        return most_similar_question, most_similar_answer

# Streamlit 앱 설정
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
        body {{
            background-color: #f5f5dc;
            font-family: 'Nanum Gothic', sans-serif;
            color: #333333;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2em;
            margin: 0 auto;
            width: 90%;
            max-width: 1200px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        }}
        .content {{
            width: 100%;
            border-radius: 10px;
            padding: 2em;
            text-align: left;
            background-color: rgba(255, 255, 255, 0.3);
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .stButton button {{
            margin: 10px 0;
            width: 100%;
            border-radius: 10px;
            border: 1px solid #0ABAB5;
            padding: 15px 30px;
            color: #0ABAB5;
            background-color: #ffffff;
            font-size: 1.2em;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }}
        .stButton button:hover {{
            background-color: #0ABAB5;
            color: #ffffff;
            cursor: pointer;
        }}
        h1 {{
            color: #0ABAB5;
            text-align: center;
            font-weight: 700;
            margin-bottom: 0.5em;
        }}
        .header {{
            font-size: 2em;
            font-weight: bold;
            color: #0ABAB5;
            text-align: center;
            margin-bottom: 1em;
        }}
        .section {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 1.2em;
            color: #333333;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            animation: fadeIn 0.5s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .small-button {{
            margin: 10px 5px;
            width: auto;
            border-radius: 5px;
            border: 1px solid #0ABAB5;
            padding: 8px 15px;
            color: #0ABAB5;
            background-color: #ffffff;
            font-size: 0.8em;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }}
        .small-button:hover {{
            background-color: #0ABAB5;
            color: #ffffff;
            cursor: pointer;
        }}
    </style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'active_button' not in st.session_state:
    st.session_state.active_button = None

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

st.title("나를 소개하는 페이지^^")

# 사이드바에 버튼 배열
st.sidebar.header("버튼을 클릭해보세요")

large_buttons = {
    "나의 아바타": "아바타.mp4",  # 이 경로를 실제 아바타 비디오 파일로 변경하세요.
    "나를 표현한 음악": "음악.mp3"  # 이 경로를 실제 음악 파일로 변경하세요.
}

small_buttons = {
    "나의 장점": "밝고 열정적이며, 사람과 이야기 하는 것을 좋아합니다.",
    "희망 진로": "물리치료사가 되고 싶습니다.",
    "좋아하는 것": "운동, 사람, 과학을 좋아합니다.",
    "싫어하는 것": " 해결할 수 없는 어려움을 겪는 것 싫어합니다.",
    "자기 소개": "저는 여러 경험을 하며 정말 내가 좋아하는 것이 무엇인지 구체적인 답을 찾고 싶습니다. 그래서 혼자서 경험해 보기 어려운 코딩을 다룬 동아리에 들어오게 되었고 앞으로도 저의 새로운 도전을 이어 나가고 싶습니다. ",
    "진로 준비": "저는 현재 안정적인 직업인 물리치료사라는 꿈을 위해 근육과 뼈를 이해하기 위해 관련 책들을 자주 보고 대학에 들어가기 위해 정보를 찾으며 물리치료사가 되기 위한 기틀을 마련하고 있습니다.",
    "취미 활동": "저는 운이 좋게도 체육을 좋아하고 잘해서 저의 취미를 즐기는 것이 체력소모가 큰 물리치료사가 되기 전 체력 증진을 할 수 있는 방법이 되어주었습니다. 또 운동은 사람의 신체를 잘 알고 움직임을 파악하는데 있어서 제 직업에 큰 도움을 줄 수 있습니다. ",
    "성공 사례": "저는 중학교 때부더 여러 표창장과 공로상을 받아왔으며. 매년 학급임원으로 활동 했습니다. 또한 여러 동아리를 만들고 교내과학토론대회, 실용음악대회, 배구경기, 캠페인 등 다양한 활동에 참여하였습니다.   "
}

for button, content in large_buttons.items():
    if st.sidebar.button(button, key=button):
        st.session_state.active_button = button if st.session_state.active_button != button else None

if st.session_state.active_button == "나의 아바타":
    st.video("아바타.mp4", format="video/mp4", start_time=0)

if st.session_state.active_button == "나를 표현한 음악":
    st.audio("음악.mp3", format="audio/mp3")

for button, content in small_buttons.items():
    if st.sidebar.button(button, key=button):
        st.session_state.active_button = button if st.session_state.active_button != button else None
    if st.session_state.active_button == button:
        st.markdown(f"<div class='section'>{content}</div>", unsafe_allow_html=True)


# 유사도 임계값 슬라이더 추가
threshold = st.slider("유사도 임계값", 0.0, 1.0, 0.43)

# 사용자 입력을 받는 입력 창
user_input = st.text_input("질문을 입력하세요:")

# 검색 버튼
if st.button("검색", key="search", help="small"):
    if user_input:
        # 유사 질문 찾기
        similar_question, answer = get_most_similar_question(user_input, threshold)
        
        if similar_question:
            st.session_state.conversation_history.append({"role": "assistant", "content": f"유사한 질문: {similar_question}"})
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
            st.write(f"**유사한 질문:** {similar_question}")
            st.write(f"**답변:** {answer}")
        else:
            st.write("유사한 질문을 찾을 수 없습니다.")

# 이전 대화 보기 버튼
if st.button("이전 대화 보기", key="view_history", help="small"):
    st.write("### 대화 기록")
    for msg in st.session_state.conversation_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        st.write(f"**{role}:** {msg['content']}")

# 새 검색 시작 버튼
if st.button("새 검색 시작", key="new_search", help="small"):
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
    st.experimental_rerun()  # 페이지를 새로고침하여 대화 기록을 초기화

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
