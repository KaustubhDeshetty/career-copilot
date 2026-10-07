import { z } from "zod";

import { MAX_TASK_CHARS } from "./limits";

// Validated in the Node layer before anything reaches Python or the LLM.
// A full job posting fits easily within MAX_TASK_CHARS, but a 50-page paste
// can't burn through the Groq token budget.

export const runRequestSchema = z.object({
  task: z
    .string()
    .trim()
    .min(3, "Type a request first.")
    .max(MAX_TASK_CHARS, `Keep it under ${MAX_TASK_CHARS.toLocaleString()} characters.`),
  runAnyway: z.boolean().optional().default(false),
});

export const saveApplicationSchema = z.object({
  company: z.string().trim().min(1).max(200),
  role: z.string().trim().min(1).max(200),
  fit_score: z.string().max(100).default("Unknown"),
  recommendation: z.string().max(2000).default("Unknown"),
  approved: z.boolean().default(false),
  revision_count: z.number().int().min(0).max(10).default(0),
});

export const saveInterviewSchema = z.object({
  topic: z.string().trim().min(1).max(300),
  difficulty: z.enum(["easy", "medium", "hard"]).default("medium"),
  num_questions: z.number().int().min(0).max(50),
  approved: z.boolean().default(false),
  revision_count: z.number().int().min(0).max(10).default(0),
});

export function firstIssue(error: z.ZodError): string {
  return error.issues[0]?.message ?? "Invalid request.";
}
