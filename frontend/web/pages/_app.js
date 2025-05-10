import { ChakraProvider, extendTheme } from '@chakra-ui/react';
import ErrorBoundary from '../components/ErrorBoundary';

const theme = extendTheme({
  fonts: {
    heading: `'Orbitron', sans-serif`,
    body: `'Inter', sans-serif`,
  },
  colors: {
    neonCyan: '#00FFFF',
    neonPurple: '#BB00FF',
  },
});

function MyApp({ Component, pageProps }) {
  return (
    <ChakraProvider theme={theme}>
      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@400;600&display=swap');
      `}</style>
      <ErrorBoundary>
        <Component {...pageProps} />
      </ErrorBoundary>
    </ChakraProvider>
  );
}

export default MyApp;
