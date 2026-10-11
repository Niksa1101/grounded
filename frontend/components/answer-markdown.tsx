import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Model text is untrusted. Raw HTML is never rendered (no rehype-raw), and links and images are
// dropped with their text kept: the only clickable links on the page are the source cards, whose
// URLs come from the database (Tech §12). The server strips URLs from the answer too, but that is a
// regex; this is the guarantee.
export function AnswerMarkdown({ markdown }: { markdown: string }) {
  return (
    <div className="space-y-3 leading-7 [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:font-mono [&_code]:text-[0.9em] [&_li]:ml-5 [&_ol]:list-decimal [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-muted [&_pre]:p-3 [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_table]:text-sm [&_td]:border [&_td]:px-2 [&_th]:border [&_th]:px-2 [&_ul]:list-disc">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        disallowedElements={["a", "img"]}
        unwrapDisallowed
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
