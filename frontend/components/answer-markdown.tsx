import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import { CitationButton } from "@/components/citation-button";
import { remarkCitationButtons } from "@/lib/citations";

// Model text is untrusted. Raw HTML is never rendered (no rehype-raw), and links and images are
// dropped with their text kept: the only clickable links on the page are the source cards, whose
// URLs come from the database (Tech §12). The server strips URLs from the answer too, but that is a
// regex; this is the guarantee.
//
// `[n]` markers that name a source become buttons (lib/citations.ts). rehype-highlight only adds
// token classes to fenced code that names a language; it never inserts HTML from the text.
export function AnswerMarkdown({
  markdown,
  sources = [],
  onCite,
}: {
  markdown: string;
  /** The display numbers of the source cards; a marker not in this list stays plain text. */
  sources?: readonly number[];
  onCite?: (n: number) => void;
}) {
  const plugins = [remarkGfm, remarkCitationButtons(new Set(sources))];
  return (
    <div className="space-y-3 leading-7 [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:font-mono [&_code]:text-[0.9em] [&_li]:ml-5 [&_ol]:list-decimal [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-muted [&_pre]:p-3 [&_pre]:text-sm [&_pre]:leading-6 [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_table]:text-sm [&_td]:border [&_td]:px-2 [&_th]:border [&_th]:px-2 [&_ul]:list-disc">
      <ReactMarkdown
        remarkPlugins={plugins}
        rehypePlugins={[rehypeHighlight]}
        disallowedElements={["a", "img"]}
        unwrapDisallowed
        components={{
          // Only the citation plugin produces a <button>: model HTML is text, not elements.
          button: ({ node }) => {
            const n = Number(node?.properties?.dataCite);
            return <CitationButton n={n} onCite={onCite} />;
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
