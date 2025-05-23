import {
  Client,
  Message,
  OmitPartialGroupDMChannel,
  TextChannel,
} from "discord.js";
import { CustomEventEmitter } from "../utils/custom-event-emitter.js";
import { Llama } from "../llama/llama.js";

type DiscordBotEvents = {
  botStarted: {};
};

type LlamaInChannel = {
  channelId: string;
  llama: Llama;
  isTyping: boolean;
  typingDebounceTimer?: NodeJS.Timeout;
};

export class DiscordBot extends CustomEventEmitter<DiscordBotEvents> {
  private readonly client: Client;
  private readonly token: string;
  private readonly clientId: string;

  private readonly llamas: LlamaInChannel[];

  constructor(token: string, clientId: string) {
    super();
    this.token = token;
    this.clientId = clientId;
    this.client = new Client({
      intents: [(0x3317efd | 0x2 | 0x8000) & ~(65536 | 64)],
    }); // ALL_UNPRIVILEGED + GUILD_MEMBERS + MESSAGE_CONTENT - GUILD_SCHEDULED_EVENTS - GUILD_INVITES
    this.llamas = [];

    this.sendIsTyping = this.sendIsTyping.bind(this);
    this.onMessageCreated = this.onMessageCreated.bind(this);
  }

  private async sendIsTyping(llama: LlamaInChannel) {
    if (!llama.isTyping) {
      const channel = this.client.channels.cache.get(llama.channelId);
      await (channel as TextChannel).sendTyping();
      llama.isTyping = true;
    }

    if (!llama.typingDebounceTimer) {
      clearTimeout(llama.typingDebounceTimer);
      llama.typingDebounceTimer = setTimeout(async () => {
        llama.isTyping = false;
        llama.typingDebounceTimer = undefined;
      }, 8000);
    }
  }

  private async onMessageCreated(
    message: OmitPartialGroupDMChannel<Message<boolean>>
  ) {
    if (message.author.id === this.client.user?.id) {
      return;
    }

    // ford's channel, dru's channel, ramp's spam channel, rimuru-dev, arthur's channel, hart's channel
    if (
      ![
        "1322487628120985640",
        "1323093263875309609",
        // "945056978126913606",
        "1323149674445541447",
        "1323511616372867122",
        "1324813950616862750",
      ].includes(message.channelId)
    ) {
      return;
    }

    let channelLlama = this.llamas.find(
      (x) => x.channelId === message.channelId
    );
    if (!channelLlama) {
      channelLlama = {
        channelId: message.channelId,
        llama: new Llama(message.channelId),
        isTyping: false,
        typingDebounceTimer: undefined,
      };
      channelLlama.llama.on("messageInProgress", () =>
        this.sendIsTyping(channelLlama!)
      );
      this.llamas.push(channelLlama);
    }

    // Determine whether we should respond
    const shouldRespond = await channelLlama.llama.shouldRespond(
      message.content,
      message.author.displayName,
      "This is a placeholder personality. Remind me to change it"
    );

    // Save the incoming message as a memory
    await channelLlama.llama.saveIncomingPrompt(
      message.content,
      message.author.displayName
    );

    // Respond, if it was determined we should do so
    if (shouldRespond) {
      const reply = await channelLlama.llama.runPrompt(message.content);

      await this.sendToChannel(reply, message.channelId);
      channelLlama.isTyping = false;
    }

    // const reply = await channelLlama.llama.runPrompt(
    //   message.content,
    //   message.author.displayName
    // );
    // if (reply && reply !== "no") {
    //   await this.sendToChannel(reply, message.channelId);
    //   channelLlama.isTyping = false;
    // }
  }

  public async start() {
    // Alert when logged in
    this.client.on("ready", () => {
      console.log(`Logged in as ${this.client.user?.tag}!`);
      this.emit("botStarted", {});
    });

    this.client.on("messageCreate", this.onMessageCreated);

    this.client.login(this.token);
  }

  private async sendToChannel(text: string, channelId: string) {
    const channel = this.client.channels.cache.get(channelId);
    await (channel as TextChannel).send(text);
  }
}
