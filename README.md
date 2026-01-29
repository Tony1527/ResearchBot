# ResearchBot
Scientific literature agent with local PDF retrieval and arXiv auto-summarization

# Usage
``` python
    from ResearchBot import *

    ## create a new Research bot
    from dotenv import load_dotenv
    load_dotenv("./local_env.env")
    bot = ResearchBot(ResearchBotConfig())
    await bot.aadd_dir("./path/to/local/dir")

    ## print result in a streaming mode
    display_system_prompt = (
        "You are a research assistant. "
        "When displaying search results, DO NOT summarize or rewrite specific paper entries. "
        "Display the papers exactly as returned by the tool, preserving the markdown structure."
    )

    msg = [SystemMessage(content=display_system_prompt)]
    msg += [HumanMessage("search 10 papers about 'graphene' in the cond-mat or quant-ph categories in the last two weeks")]
    await bot.stream_response(input=msg)
```