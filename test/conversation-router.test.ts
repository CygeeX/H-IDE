import { describe, expect, it, vi } from "vitest";
import { ConversationRouter, conversationRouterInternals } from "../src/conversation/router.js";

const signal = () => new AbortController().signal;

describe("Assistant V2 conversation router", () => {
  it("answers obvious greetings without a model call or a mission", async () => {
    const agent = { generate: vi.fn() };
    const result = await new ConversationRouter(agent as never).route({
      conversationId: "conversation-1",
      locale: "zh-CN",
      text: "你好！",
    }, signal());

    expect(result).toMatchObject({
      intent: "SOCIAL_CHAT",
      needsRobot: false,
      shouldCreateMission: false,
    });
    expect(result.assistantMessage).toContain("你好");
    expect(agent.generate).not.toHaveBeenCalled();
  });

  it("answers bounded arithmetic locally with precedence and parentheses", async () => {
    const agent = { generate: vi.fn() };
    const result = await new ConversationRouter(agent as never).route({
      conversationId: "conversation-1",
      locale: "zh-CN",
      text: "请帮我计算 (12 + 3) × 2 等于多少？",
    }, signal());

    expect(result).toMatchObject({
      intent: "GENERAL_QA",
      assistantMessage: "结果是 30。",
      needsRobot: false,
      shouldCreateMission: false,
    });
    expect(agent.generate).not.toHaveBeenCalled();
    expect(conversationRouterInternals.evaluateArithmetic("10 / 0")).toBeUndefined();
  });

  it("keeps mixed greeting/action requests on the model path", async () => {
    const agent = {
      generate: vi.fn().mockResolvedValue({
        object: {
          intent: "ROBOT_ACTION",
          assistantMessage: "我来执行。",
          clarification: null,
        },
      }),
    };
    const result = await new ConversationRouter(agent as never).route({
      conversationId: "conversation-1",
      robotId: "g1-sim",
      locale: "zh-CN",
      text: "你好，向前走两米",
    }, signal());

    expect(agent.generate).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      intent: "ROBOT_ACTION",
      needsRobot: true,
      shouldCreateMission: true,
    });
    expect(result.assistantMessage).toContain("待审核");
    expect(result.assistantMessage).toContain("不会在这里直接执行");
  });

  it("passes bounded history to the model with low-latency provider settings", async () => {
    const agent = {
      generate: vi.fn().mockResolvedValue({
        object: {
          intent: "GENERAL_QA",
          assistantMessage: "上一步的答案是四。",
          clarification: null,
        },
      }),
    };
    const result = await new ConversationRouter(agent as never).route({
      conversationId: "conversation-2",
      locale: "zh-CN",
      text: "那上一步的答案呢？",
      summary: "用户正在讨论一道算术题。",
      history: [
        { role: "user", content: "二加二是多少？" },
        { role: "assistant", content: "我可以帮你计算。" },
      ],
    }, signal());

    expect(result.intent).toBe("GENERAL_QA");
    const [prompt, options] = agent.generate.mock.calls[0]!;
    expect(prompt).toContain("用户正在讨论一道算术题");
    expect(prompt).toContain("二加二是多少");
    expect(options).toMatchObject({
      modelSettings: { temperature: 0, reasoning: "none" },
      providerOptions: {
        alibaba: { enableThinking: false },
        deepseek: { thinking: { type: "disabled" } },
      },
    });
  });

  it("makes observation non-motion and blocks mission creation while clarifying", async () => {
    const observationAgent = {
      generate: vi.fn().mockResolvedValue({
        object: {
          intent: "ROBOT_OBSERVE",
          assistantMessage: "我看看。",
          clarification: null,
        },
      }),
    };
    const observation = await new ConversationRouter(observationAgent as never).route({
      conversationId: "conversation-3",
      robotId: "g1-sim",
      locale: "zh-CN",
      text: "只观察当前画面里有什么，不要移动",
    }, signal());
    expect(observation).toMatchObject({
      intent: "ROBOT_OBSERVE",
      needsRobot: true,
      shouldCreateMission: false,
    });
    expect(observation.assistantMessage).toContain("不会移动机器人");

    const clarifyAgent = {
      generate: vi.fn().mockResolvedValue({
        object: {
          intent: "ROBOT_ACTION",
          assistantMessage: "我需要确认目标。",
          clarification: "你想让机器人去哪个房间？",
        },
      }),
    };
    const clarification = await new ConversationRouter(clarifyAgent as never).route({
      conversationId: "conversation-4",
      locale: "zh-CN",
      text: "让机器人过去",
    }, signal());
    expect(clarification).toMatchObject({
      intent: "ROBOT_ACTION",
      needsRobot: true,
      shouldCreateMission: false,
      clarification: "你想让机器人去哪个房间？",
    });
  });

  it.each([
    ["SOCIAL_CHAT", false, false],
    ["GENERAL_QA", false, false],
    ["ROBOT_OBSERVE", true, false],
    ["ROBOT_ACTION", true, true],
    ["SCENE_TEACH", true, false],
    ["ADMIN_DEV", false, false],
  ] as const)(
    "derives authority flags for %s instead of trusting the model",
    async (intent, needsRobot, shouldCreateMission) => {
      const agent = {
        generate: vi.fn().mockResolvedValue({
          object: { intent, assistantMessage: "已分类。", clarification: null },
        }),
      };
      const result = await new ConversationRouter(agent as never).route({
        conversationId: `conversation-${intent}`,
        locale: "zh-CN",
        text: "这是一条需要模型分类的上下文请求",
      }, signal());
      expect(result).toMatchObject({ intent, needsRobot, shouldCreateMission });
    },
  );
});
