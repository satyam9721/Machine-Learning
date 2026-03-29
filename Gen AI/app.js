import Groq from "groq-sdk";

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

async function main() {
  const completion = await groq.chat.completions.create({
    temperature:1, //ranege 0-2, higher values means more creative response
    model: "openai/gpt-oss-20b",
    messages: [
        {
            //behave define of model
        role: "system",
        content: "You are Jarvis, a helpful assistant. Be always polite",
      },
      {
        role: "user",
        content: "Hi",
      },
    ],
  });

  console.log(completion.choices[0].message.content);
}

main();
