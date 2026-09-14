from src.travel_assistant.router import classify


def test_weather_question_selects_weather_only():
    intent = classify("What is the weather in Singapore for the next three days?", destination_currency="SGD")
    assert intent.needs_weather is True
    assert intent.needs_currency is False


def test_currency_question_selects_currency_only():
    intent = classify("Convert INR 60,000 to SGD.", destination_currency="SGD")
    assert intent.needs_weather is False
    assert intent.needs_currency is True
    assert intent.currency.amount == 60000
    assert intent.currency.from_currency == "INR"
    assert intent.currency.to_currency == "SGD"


def test_combined_question_selects_both_tools():
    intent = classify(
        "Convert INR 60,000 to SGD and adjust my Singapore itinerary if rain is expected.",
        destination_currency="SGD",
    )
    assert intent.needs_weather is True
    assert intent.needs_currency is True
