import assert from "node:assert/strict";
import test from "node:test";
import { ESLint } from "eslint";

const examples = [
  ["react/display-name", "export default () => <div />;", 2],
  ["jsx-a11y/alt-text", 'export default function Photo() { return <img src="/image.jpg" />; }', 1],
  ["@typescript-eslint/no-explicit-any", "export function identity(value: any) { return value; }", 2],
];

for (const [rule, source, severity] of examples) {
  test(`the project lint config still detects ${rule}`, async () => {
    const [result] = await new ESLint().lintText(source, {
      filePath: "components/LintProbe.tsx",
    });
    assert.equal(result.fatalErrorCount, 0, "lint plugins must load and parse TypeScript");
    assert.ok(result.messages.some((message) => message.ruleId === rule && message.severity === severity));
  });
}
