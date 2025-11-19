from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Google 임베딩 테스트
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key="AIzaSyAZLJF6oW0xQsyXxaPKqyvUNrJHnVIi-2Y"
)

test_text = "테스트 문장입니다"
embedding = embeddings.embed_query(test_text)

print(f"임베딩 차원: {len(embedding)}")
print(f"임베딩 타입: {type(embedding)}")
print(f"첫 5개 값: {embedding[:5]}")
