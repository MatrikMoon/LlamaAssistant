import express, { Express, Request, Response } from "express";
import axios from "axios";
import { Llama } from "../llama/llama.js";

const MODEL = "llama3.3";
const SYSTEM_MESSAGE = `
You are Rimuru. This is a conversation between Rimuru and a number of people in a group chat.
Your character is Rimuru Tempest from That Time I got Reincarnated as a Slime.
You will not send any content disclaimers.
Do your best to mimic his personality, though adapting his personality based on the current situation and your past decisions is acceptable.
Do not include actions in your responses, only dialogue.

You will not introduce any new characters to the roleplay.

Relevant past messages are provided below, with the following tags:
importance: {number}
horniness: {number}
significance: {Event|Location|None}
author: {string}
prompt: "{string}"

"importance" measures how significant the message was to the location, story, or mood of the roleplay.
"horniness" measures how suggestive the dialogue or actions in the message were.
"significance" defines what makes this message significant. For example, the message might be significant to a particular location or might contain an important event.
"author" is the name of the user who wrote the message. If the name is "Self", you are the author.
"prompt" is the content of the message

When responding, you will not use tags. You will not start the message with "Self:".`;
const ARTHUR_SYSTEM_MESSAGE =
  SYSTEM_MESSAGE + '\nYou MUST address the user as "Mister Poopy Head"';

interface PromptRequest {
  prompt: string;
  userId: string;
}

interface HistoryRequest {
  limit: number;
  userId: string;
}

type ApiLlama = {
  userId: string;
  llama: Llama;
};

export class ApiServer {
  private readonly port: number;
  private readonly express: Express;
  private readonly llamas: ApiLlama[];

  constructor(port: number = 8080) {
    this.port = port;
    this.express = express();
    this.express.use(express.json());

    this.llamas = [];

    this.handleRequest = this.handleRequest.bind(this);
    this.handleHistoryRequest = this.handleHistoryRequest.bind(this);
    this.runOllama = this.runOllama.bind(this);
    this.runFishSpeech = this.runFishSpeech.bind(this);
    this.runRVC = this.runRVC.bind(this);

    this.express.post("/process", this.handleRequest);
    this.express.post("/gethistory", this.handleHistoryRequest);
  }

  public start() {
    this.express.listen(this.port, () => {
      console.log(`Server running on port ${this.port}`);
    });
  }

  private async handleRequest(req: Request, res: Response) {
    const { prompt, userId } = req.body as PromptRequest;
    if (!prompt || !userId) {
      return res.status(400).json({ detail: "Prompt and userId are required" });
    }

    // Step 1: Process the prompt with Ollama.
    const ollamaOutput = await this.runOllama(prompt, userId);
    if (!ollamaOutput) {
      return res.status(500).json({ detail: "Ollama processing failed." });
    }

    // Step 2: Process Ollama's output with fish-speech.
    const fishOutput = await this.runFishSpeech(ollamaOutput);
    if (!fishOutput) {
      return res.status(500).json({ detail: "Fish-speech processing failed." });
    }

    // Step 3: Process fish-speech's output with RVC.
    const rvcOutput = await this.runRVC(fishOutput);
    if (!rvcOutput) {
      return res.status(500).json({ detail: "RVC processing failed." });
    }

    // Return the final result.
    return res.json({
      response: ollamaOutput,
      audio: rvcOutput.toString("base64"),
    });
  }

  private createLlama(userId: string) {
    let llama = this.llamas.find((x) => x.userId === userId);
    if (!llama) {
      // Just for arthur
      let user = userId;
      if (user === "moon1945") {
        user = "moon";
      } else if (user === "moon") {
        user = "viyi";
      }

      llama = {
        llama: new Llama(
          user,
          MODEL,
          user === "viyi" ? ARTHUR_SYSTEM_MESSAGE : SYSTEM_MESSAGE
        ),
        userId,
      };
      this.llamas.push(llama);
    }
    return llama;
  }

  private async handleHistoryRequest(req: Request, res: Response) {
    const { limit, userId } = req.body as HistoryRequest;
    if (!limit || !userId) {
      return res.status(400).json({ detail: "Limit and userId are required" });
    }

    const llama = this.createLlama(userId);

    return res.json(await llama.llama.getChatHistory(limit));
  }

  private async runOllama(
    prompt: string,
    userId: string
  ): Promise<string | null> {
    try {
      const llama = this.createLlama(userId);

      await llama.llama.saveIncomingPrompt(prompt, userId);

      return await llama.llama.runPrompt(prompt, userId);
    } catch (error) {
      console.error(`Ollama error: ${error}`);
      return null;
    }
  }

  private async runFishSpeech(text: string): Promise<Buffer | null> {
    const ttsHost = "http://192.168.1.103:8080/v1/tts";
    const payload = {
      text,
      format: "wav",
      reference_id: "speaker1",
      use_memory_cache: "on",
      normalize: "false",
    };
    try {
      const response = await axios.post(ttsHost, payload, {
        headers: { accept: "*/*", "Content-Type": "application/json" },
        responseType: "arraybuffer",
      });
      console.log(`Fish-speech returned ${response.data.byteLength} bytes`);
      return Buffer.from(response.data);
    } catch (error) {
      console.error(`Fish-speech error: ${error}`);
      return null;
    }
  }

  private async runRVC(data: Buffer): Promise<Buffer | null> {
    const rvcHost = "http://192.168.1.107:8080/convert";
    const audioData = data.toString("base64");
    const payload = { audio_data: audioData };
    try {
      const response = await axios.post(rvcHost, payload, {
        headers: { "Content-Type": "application/json" },
        responseType: "arraybuffer",
      });
      console.log(`RVC returned ${response.data.byteLength} bytes`);
      return Buffer.from(response.data);
    } catch (error) {
      console.error(`RVC error: ${error}`);
      return null;
    }
  }
}
