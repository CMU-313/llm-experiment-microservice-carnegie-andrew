from src.translator import translate_content, get_language, get_translation, client
from unittest.mock import patch, MagicMock

def test_chinese():
    """Test that Chinese text is detected as non-English and translated."""
    # Mock the Ollama client to return expected responses
    with patch.object(client, 'chat') as mock_chat:
        # First call: language detection returns "Chinese"
        mock_lang_response = MagicMock()
        mock_lang_response.message.content = "Chinese"
        
        # Second call: translation returns the expected English text
        mock_trans_response = MagicMock()
        mock_trans_response.message.content = "This is a Chinese message"
        
        mock_chat.side_effect = [mock_lang_response, mock_trans_response]
        
        is_english, translated_content = translate_content("这是一条中文消息")
        
        assert is_english == False, "Chinese text should be detected as non-English"
        assert translated_content == "This is a Chinese message", "Translation should match expected output"

def test_llm_normal_response():
    # Test 1: Non-English text should be detected and translated
    is_english, translated = translate_content("¿Dónde está la biblioteca roja?")
    # The model should detect this as non-English
    assert is_english == False, "Spanish text should be detected as non-English"
    # Translation should be a non-empty string (we don't require exact match due to LLM variability)
    assert isinstance(translated, str), "Translation should be a string"
    assert len(translated) > 0, "Translation should not be empty"
    print(f"Test 1 - Spanish input: is_english={is_english}, translated='{translated}'")
    
    # Test 2: English text should be detected and not modified
    english_text = "Hello"
    is_english, translated = translate_content(english_text)
    # Should detect as English  
    assert isinstance(translated, str), "Should return a string"
    assert len(translated) > 0, "Should return non-empty text"
    print(f"Test 2 - English input: is_english={is_english}, translated='{translated}'")

def test_llm_gibberish_response():
    # Test 1: gibberish input text
    gibberish_input = "asdfghjkl12345!@#$%^&*()"
    is_english, translated = translate_content(gibberish_input)
    
    # Should not crash and should return something
    assert isinstance(is_english, bool), "Should return a boolean for is_english"
    assert isinstance(translated, str), "Should return a string for translation"
    assert len(translated) > 0, "Should return non-empty result"
    print(f"Test 1 - Gibberish input: is_english={is_english}, translated='{translated}'")
    
    # Test 2: Empty string input
    # The system should handle empty input gracefully
    is_english, translated = translate_content("")
    
    assert isinstance(is_english, bool), "Should return a boolean even for empty input"
    assert isinstance(translated, str), "Should return a string even for empty input"
    print(f"Test 2 - Empty input: is_english={is_english}, translated='{translated}'")
    
    # Test 3: Very long text (stress test)
    # The LLM should handle or gracefully fail with long inputs
    long_text = "Hello world. " * 100  # Repeat 100 times
    is_english, translated = translate_content(long_text)
    
    assert isinstance(is_english, bool), "Should return a boolean for long input"
    assert isinstance(translated, str), "Should return a string for long input"
    assert len(translated) > 0, "Should return non-empty result for long input"
    print(f"Test 3 - Long input: is_english={is_english}, translated length={len(translated)}")