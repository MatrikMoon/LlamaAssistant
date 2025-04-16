export function defaultTool() {
  console.log("Default!");
}

export const defaultToolTool = {
  type: "function",
  function: {
    name: "defaultTool",
    description:
      "This is the default tool, call this tool when none of the other tools seem to fit the users request",
    parameters: {
      type: "object",
      required: [],
      properties: {},
    },
  },
};
