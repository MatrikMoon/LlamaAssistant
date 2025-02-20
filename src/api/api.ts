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
  inputDebounceTimer?: NodeJS.Timeout;
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
    this.handleVoiceRequest = this.handleVoiceRequest.bind(this);
    this.handleHistoryRequest = this.handleHistoryRequest.bind(this);
    this.runOllama = this.runOllama.bind(this);
    this.runFishSpeech = this.runFishSpeech.bind(this);
    this.runRVC = this.runRVC.bind(this);

    this.express.post("/process", this.handleRequest);
    this.express.post("/processVoice", this.handleVoiceRequest);
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

    const ollamaOutput = await this.runOllama(prompt, userId);
    if (!ollamaOutput) {
      return res.status(500).json({ detail: "Ollama processing failed." });
    }

    let ollamaOutputWithoutThink = ollamaOutput;
    if (ollamaOutput.includes("</think>")) {
      ollamaOutputWithoutThink = ollamaOutput.substring(
        ollamaOutput.indexOf("</think>") + "</think>".length
      );
    }

    return await this.convertTextToAudioAndSendResponse(
      ollamaOutputWithoutThink,
      res
    );
  }

  private async handleVoiceRequest(req: Request, res: Response) {
    const { prompt, userId } = req.body as PromptRequest;
    if (!prompt || !userId) {
      return res.status(400).json({ detail: "Prompt and userId are required" });
    }

    const filteredPrompt = this.filterWordsFromSTT(prompt);

    const llama = this.createLlama(userId);

    if (llama.inputDebounceTimer) {
      await llama.llama.saveIncomingPrompt(prompt);
      clearTimeout(llama.inputDebounceTimer);
    }

    llama.inputDebounceTimer = setTimeout(async () => {
      llama.inputDebounceTimer = undefined;

      const shouldRespond = await llama.llama.shouldRespond(
        filteredPrompt,
        userId
      );

      await llama.llama.saveIncomingPrompt(prompt);

      if (shouldRespond) {
        const ollamaOutput = await this.runOllama(
          filteredPrompt,
          userId,
          false
        );
        if (!ollamaOutput) {
          return res.status(500).json({ detail: "Ollama processing failed." });
        }

        let ollamaOutputWithoutThink = ollamaOutput;
        if (ollamaOutput.includes("</think>")) {
          ollamaOutputWithoutThink = ollamaOutput.substring(
            ollamaOutput.indexOf("</think>") + "</think>".length
          );
        }

        await this.convertTextToAudioAndSendResponse(
          ollamaOutputWithoutThink,
          res
        );
      } else {
        return res
          .status(204)
          .json({ detail: "Ollama determined not to respond" });
      }
    }, 2000);
  }

  private async handleHistoryRequest(req: Request, res: Response) {
    const { limit, userId } = req.body as HistoryRequest;
    if (!limit || !userId) {
      return res.status(400).json({ detail: "Limit and userId are required" });
    }

    const llama = this.createLlama(userId);

    return res.json(await llama.llama.getChatHistory(limit));
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

  private filterWordsForTTS(text: string) {
    return text
      .replace("rimuru", "reemaru")
      .replace("Rimuru", "Reemaru")
      .replace(" - ", ", ");
  }

  private filterWordsFromSTT(text: string) {
    return text
      .replace(" Reamer", " Rimuru")
      .replace(" reamer", " Rimuru")
      .replace("Remaru", "Rimuru")
      .replace("remaru", "Rimuru")
      .replace("Remerow", "Rimuru")
      .replace("remerow", "Rimuru")
      .replace("Reemaru", "Rimuru")
      .replace("reemaru", "Rimuru")
      .replace("Reemuru", "Rimuru")
      .replace("reemuru", "Rimuru")
      .replace("Imaru", "Rimuru")
      .replace("imaru", "Rimuru");
  }

  private async convertTextToAudioAndSendResponse(text: string, res: Response) {
    const fishOutput = await this.runFishSpeech(this.filterWordsForTTS(text));
    if (!fishOutput) {
      return res.status(500).json({ detail: "Fish-speech processing failed." });
    }

    const rvcOutput = await this.runRVC(fishOutput);
    if (!rvcOutput) {
      return res.status(500).json({ detail: "RVC processing failed." });
    }

    return res.json({
      response: text,
      audio: rvcOutput.toString("base64"),
    });
  }

  private async runOllama(
    prompt: string,
    userId: string,
    saveIncomingPrompt: boolean = true
  ): Promise<string | null> {
    try {
      const llama = this.createLlama(userId);

      if (saveIncomingPrompt) {
        await llama.llama.saveIncomingPrompt(prompt, userId);
      }

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
