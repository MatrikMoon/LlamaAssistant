import { DiscordBot } from "./discord/discordService.js";
import { config } from "./config.js";

const discordBot = new DiscordBot(config.DISCORD_TOKEN, config.DISCORD_CLIENT_ID);
await discordBot.start();