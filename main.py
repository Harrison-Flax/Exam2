from tasks import get_response, text_summarization, code_generator, writer, multilingual
from benchmarks import write_log

# Three Models: TinyLLama. TinyDolphin, Reader-lm
models = ['tinyllama', 'tinydolphin', 'reader-lm', 'DropAI']

def main():
    # Stored questions separately
    questions = [
        'What is the computational power needed to solve the hardest math problem in the world?',
        'What is the most advanced math problem a calculator can solve?',
        'Can computers read math symbols effectively and easily without assistance?'
    ]

    # Answer the questions
    for question in questions:
        print(f"Question: {question}\n")
        for model in models:
            response = get_response(model, question)
            print(f"Response from {model.capitalize()}:\n{response}\n")

    text_input = """
        Jewish delis have long been an emblem of symbol of kosher cuisine, but also one of Ashkenazi roots in Eastern Europe
        and particularly in Europe. Delis like Katz's and Second Avenue have been around in New York City since Jewish immigrants
        starting coming in droves to the United States in Ellis Island before and after WWII. These delis have served as a place for
        Jewish culture and cuisine to thrive while keeping the hardworking stories of Jewish immigrants alive in NYC. However, these
        eateries are now under more threat than ever as rents in Manhattan increase dramatically, food prices rise, and maintance for
        these delis are exhausting and ever more expensive as the years go by. Some of the latest victims have been the Carnegie Deli
        and Lindy's which closed under financial and landlord pressure. It is rather unfortunate that these delis have had to undertake
        such a crisis, but their preservation is more important than ever to keep the culture and history alive of the Jewish people and
        their immigrant backgrounds in NYC.
        """

    # Summarize from all 3 models
    for model in models:
        summary = text_summarization(model, text_input)
        print(f"Summary from {model.capitalize()}:\n{summary}\n")
    
    code_input = """
        Generate a simple function that can pull the latest information about the Dollar General stock
        """

    # Generate from all 3 models
    for model in models:
        code = code_generator(model, code_input)
        print(f"Code from {model.capitalize()}:\n{code}\n")

    story_input = """
        Generate a short story that follows the adventures of Sonic the Hedgehog as he falls down from grace and Shadow
        ends up replacing him. Make it funny too and don't be afraid to get wacky with it!
        """

    # Generate from all 3 models
    for model in models:
        story = writer(model, story_input)
        print(f"Here is a short story from {model.capitalize()}:\n{story}\n")
    
    lang_input = """
        Generate a paragraph in Hebrew, Japanese, and French and then translate it to English.
        Show both the paragraph in the original language and translation in English.
        """

    # Generate from all 3 models
    for model in models:
        lang = multilingual(model, lang_input)
        print(f"Here is a translation from {model.capitalize()}:\n{lang}\n")
        
    # Write the output to the .txt file
    write_log()
    
if __name__ == "__main__":
    main()