import express, { Express, Request, Response } from "express";
import axios from "axios";
import { Llama } from "../llama/llama.js";

const MODEL = "llama3.3";

interface PromptRequest {
  prompt: string;
  userId: string;

  personality?: string;
  gender?: string;
  sourceMaterial?: string;
}

interface HistoryRequest {
  limit: number;
  userId: string;
}

type ApiLlama = {
  userId: string;
  llama: Llama;
  currentlyProcessingVoice?: boolean;
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
    this.handleDeleteHistoryRequest =
      this.handleDeleteHistoryRequest.bind(this);
    this.runOllama = this.runOllama.bind(this);
    this.runFishSpeech = this.runFishSpeech.bind(this);
    this.runRVC = this.runRVC.bind(this);

    this.express.post("/process", this.handleRequest);
    this.express.post("/processVoice", this.handleVoiceRequest);
    this.express.post("/getHistory", this.handleHistoryRequest);
    this.express.post("/deleteHistory", this.handleDeleteHistoryRequest);
  }

  public start() {
    this.express.listen(this.port, () => {
      console.log(`Server running on port ${this.port}`);
    });
  }

  private async handleRequest(req: Request, res: Response) {
    const { prompt, userId, personality, gender, sourceMaterial } =
      req.body as PromptRequest;
    if (!prompt || !userId) {
      return res.status(400).json({ detail: "Prompt and userId are required" });
    }

    const ollamaOutput = await this.runOllama(
      prompt,
      userId,
      personality,
      gender,
      sourceMaterial
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

    return await this.convertTextToAudioAndSendResponse(
      prompt,
      ollamaOutputWithoutThink,
      personality ?? "Rimuru",
      res
    );
  }

  private async handleVoiceRequest(req: Request, res: Response) {
    const { prompt, userId, personality, gender, sourceMaterial } =
      req.body as PromptRequest;
    if (!prompt || !userId) {
      return res.status(400).json({ detail: "Prompt and userId are required" });
    }

    const filteredPrompt = this.filterWordsFromSTT(prompt);

    const llama = this.createLlama(userId, personality, gender, sourceMaterial);

    if (llama.inputDebounceTimer) {
      // For now, we're only saving prompts rimuru responds to
      // await llama.llama.saveIncomingPrompt(filteredPrompt);
      clearTimeout(llama.inputDebounceTimer);
      res.status(204).json({ detail: "Prompt debounced" });
    } else if (llama.currentlyProcessingVoice) {
      // For now, we're only saving prompts rimuru responds to
      // await llama.llama.saveIncomingPrompt(filteredPrompt);
      res
        .status(204)
        .json({ detail: "Will not process due to prompt in progress" });
    }

    llama.inputDebounceTimer = setTimeout(async () => {
      llama.inputDebounceTimer = undefined;
      llama.currentlyProcessingVoice = true;

      const shouldRespond = await llama.llama.shouldRespond(
        filteredPrompt,
        userId,
        personality ?? "Rimuru",
        Llama.getShouldRespondForPersonality(
          personality ?? "Rimuru",
          gender ?? "male",
          sourceMaterial ?? "That Time I got Reincarnated as a Slime"
        )
      );

      // For now, we're only saving prompts rimuru responds to
      // await llama.llama.saveIncomingPrompt(filteredPrompt);

      if (shouldRespond) {
        await llama.llama.saveIncomingPrompt(filteredPrompt, userId);

        const ollamaOutput = await this.runOllama(
          filteredPrompt,
          userId,
          personality,
          gender,
          sourceMaterial,
          false
        );
        if (!ollamaOutput) {
          llama.currentlyProcessingVoice = false;
          return res.status(500).json({ detail: "Ollama processing failed." });
        }

        let ollamaOutputWithoutThink = ollamaOutput;
        if (ollamaOutput.includes("</think>")) {
          ollamaOutputWithoutThink = ollamaOutput.substring(
            ollamaOutput.indexOf("</think>") + "</think>".length
          );
        }

        await this.convertTextToAudioAndSendResponse(
          filteredPrompt,
          ollamaOutputWithoutThink,
          personality ?? "Rimuru",
          res
        );
        llama.currentlyProcessingVoice = false;
      } else {
        llama.currentlyProcessingVoice = false;
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
    if (llama) {
      // We don't want this llama to stick around, just in case the user wasn't done typing...
      // Moon's note: this is nasty, but I'll remnid you this is only a debug interface
      this.deleteLlama(userId);
      return res.json(await llama.llama.getChatHistory(limit));
    }

    return res.status(404).json("That Llama does not exist");
  }

  private async handleDeleteHistoryRequest(req: Request, res: Response) {
    const { userId } = req.body as HistoryRequest;
    if (!userId) {
      return res.status(400).json({ detail: "UserId is required" });
    }

    const llama = this.createLlama(userId);
    if (llama) {
      await llama.llama.deleteConversation();
      this.deleteLlama(userId);

      return res.status(200).json({ detail: "Conversation was deleted" });
    }

    return res.status(404).json("That Llama does not exist");
  }

  private createLlama(
    userId: string,
    personality: string = "Rimuru",
    gender: string = "male",
    sourceMaterial: string = "That Time I got Reincarnated as a Slime"
  ) {
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
          Llama.getSystemPromptForPersonality(
            personality,
            gender,
            sourceMaterial
          )
        ),
        userId,
      };
      this.llamas.push(llama);
    }
    return llama;
  }

  private getLlama(userId: string) {
    return this.llamas.find((x) => x.userId === userId);
  }

  private deleteLlama(userId: string) {
    const index = this.llamas.findIndex((x) => x.userId === userId);
    if (index > -1) {
      this.llamas.splice(index, 1);
    }
  }

  private filterWordsForTTS(text: string) {
    return text
      .replace("Rimuru", "Reemaru")
      .replace("Shion", "Sheeown")
      .replace("*", "")
      .replace(" - ", ", ");
  }

  private filterWordsFromSTT(text: string) {
    return text
      .replace(" Reamer", " Rimuru")
      .replace(" reamer", " Rimuru")
      .replace(" Rimmer", " Rimuru")
      .replace(" rimmer", " Rimuru")
      .replace(" Reimuer", " Rimuru")
      .replace(" reimuer", " Rimuru")
      .replace(" Remaru", " Rimuru")
      .replace(" remaru", " Rimuru")
      .replace(" Remerow", " Rimuru")
      .replace(" remerow", " Rimuru")
      .replace(" Reemaru", " Rimuru")
      .replace(" reemaru", " Rimuru")
      .replace(" Reemuru", " Rimuru")
      .replace(" reemuru", " Rimuru")
      .replace(" Rimaru", " Rimuru")
      .replace(" rimaru", " Rimuru")
      .replace(" Remeru", " Rimuru")
      .replace(" remeru", " Rimuru")
      .replace(" Remer", " Rimuru")
      .replace(" remer", " Rimuru")
      .replace(" Imaru", " Rimuru")
      .replace(" imaru", " Rimuru")
      .replace(" Remaroo", " Rimuru")
      .replace(" remaroo", " Rimuru");
  }

  private async convertTextToAudioAndSendResponse(
    respondingTo: string,
    response: string,
    personality: string,
    res: Response
  ) {
    // Right now we only support Rimuru and Frieren voices
    if (
      personality !== "Rimuru" &&
      personality !== "Frieren" &&
      personality !== "Gura"
    ) {
      personality = "default";
    }

    const fishOutput = await this.runFishSpeech(
      this.filterWordsForTTS(response),
      personality
    );
    if (!fishOutput) {
      return res.status(500).json({ detail: "Fish-speech processing failed." });
    }

    const rvcOutput = await this.runRVC(fishOutput, personality);
    if (!rvcOutput) {
      return res.status(500).json({ detail: "RVC processing failed." });
    }

    return res.json({
      respondingTo,
      response,
      audio: rvcOutput.toString("base64"),
    });
  }

  private async runOllama(
    prompt: string,
    userId: string,
    personality: string = "Rimuru",
    gender: string = "male",
    sourceMaterial: string = "That Time I got Reincarnated as a Slime",
    saveIncomingPrompt: boolean = true
  ): Promise<string | null> {
    try {
      const llama = this.createLlama(
        userId,
        personality,
        gender,
        sourceMaterial
      );

      if (saveIncomingPrompt) {
        await llama.llama.saveIncomingPrompt(prompt, userId);
      }

      return await llama.llama.runPrompt(prompt, userId);
    } catch (error) {
      console.error(`Ollama error: ${error}`);
      return null;
    }
  }

  private async runFishSpeech(
    text: string,
    personality: string
  ): Promise<Buffer | null> {
    const ttsHost = "http://192.168.1.103:8080/v1/tts";
    const payload = {
      text,
      format: "wav",
      reference_id: personality.toLowerCase(),
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

  private async runRVC(
    data: Buffer,
    personality: string
  ): Promise<Buffer | null> {
    const rvcHost = "http://192.168.1.107:8080";
    const audioData = data.toString("base64");
    const payload = { audio_data: audioData };
    try {
      // Load the correct RVC model
      await axios.post(`${rvcHost}/models/${personality.toLowerCase()}`, {
        headers: { "Content-Type": "application/json" },
        responseType: "arraybuffer",
      });
      console.log(`RVC loaded model: ${personality.toLowerCase()}`);

      const response = await axios.post(`${rvcHost}/convert`, payload, {
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
