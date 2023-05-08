from project3 import preprocess_text


def test_preprocess_text():
    # Define input text and expected output
    input_text = "This is a test sentence. It contains punctuation, numbers 123, and stop words such as 'the' and 'a'."
    expected_output = "test sentence contain punctuation number stop word"

    # Preprocess the input text
    output = preprocess_text(input_text, ["a", "the"])

    # Assert that the output is equal to the expected output
    assert output == expected_output

