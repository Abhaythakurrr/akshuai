import Head from 'next/head'
import Image from 'next/image'
import styles from '../styles/Home.module.css'

export default function Home() {
  return (
    <div className={styles.container}>
      <Head>
        <title>AkshuAI Web Frontend</title>
        <meta name="description" content="AkshuAI Web Frontend" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>
          Welcome to AkshuAI!
        </h1>

        <p className={styles.description}>
          This is the web frontend.
        </p>

        {/* TODO: Add chat interface and integrate with backend services */}

      </main>

      <footer className={styles.footer}>
        Powered by AkshuAI
      </footer>
    </div>
  )
}
