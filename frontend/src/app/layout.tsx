import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Book Assistant - Deep Space Document Search",
  description: "A premium document assistant powered by Mistral AI and Chroma DB. Parse, chunk, embed, and query your PDF documents with semantic context.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
