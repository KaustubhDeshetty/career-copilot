import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Renders LLM output. GFM is needed for the Fit Assessment tables. */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}
