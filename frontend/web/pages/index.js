import { useEffect, useState } from 'react';
import {
  Box,
  Button,
  Input,
  VStack,
  Text,
  Spinner,
  useToast,
} from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { auth, db, functions } from '../lib/firebase';
import {
  signInWithPopup,
  GoogleAuthProvider,
  onAuthStateChanged,
} from 'firebase/auth';
import { collection, addDoc, onSnapshot } from 'firebase/firestore';
import { httpsCallable } from 'firebase/functions';
import dynamic from 'next/dynamic';
import Starfield from '../components/Starfield';
import Logo from '../components/Logo';

const WebFont = dynamic(() => import('webfontloader'), { ssr: false });

export default function Home() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const toast = useToast();

  useEffect(() => {
    if (WebFont && typeof WebFont.load === 'function') {
      WebFont.load({
        google: {
          families: ['Orbitron:400,700', 'Inter:400,600'],
        },
      });
    }
  }, []);

  useEffect(() => {
    const unsubscribeAuth = onAuthStateChanged(auth, (user) => {
      setUser(user);
    });

    if (user) {
      const unsubscribeMemory = onSnapshot(
        collection(db, 'memory'),
        (snapshot) => {
          snapshot.docChanges().forEach((change) => {
            if (
              change.type === 'added' &&
              change.doc.data().user_id === user.uid
            ) {
              setResponse(change.doc.data().value);
            }
          });
        }
      );
      return () => unsubscribeMemory();
    }

    return () => unsubscribeAuth();
  }, [user]);

  const handleLogin = async () => {
    try {
      const provider = new GoogleAuthProvider();
      const result = await signInWithPopup(auth, provider);
      setUser(result.user);
      toast({
        title: 'Logged in successfully.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Login error:', error);
      toast({
        title: 'Login failed.',
        description: error.message,
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    }
  };

  const handleGenerate = async () => {
    if (!prompt || !user) return;
    setLoading(true);
    try {
      const generateText = httpsCallable(functions, 'generateText');
      const result = await generateText({
        text: prompt,
        user_id: user.uid,
        session_id: 'default',
      });
      const data = result.data;
      setResponse(data.final_response);

      await addDoc(collection(db, 'memory'), {
        user_id: user.uid,
        session_id: 'default',
        key: 'generated_text',
        value: data.final_response,
        timestamp: new Date(),
      });
    } catch (error) {
      console.error('Error:', error);
      setResponse('Error generating response');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      minH="100vh"
      bg="linear-gradient(135deg, #0A0A1E 0%, #1a1a2e 100%)"
      color="white"
      fontFamily={'"Inter", sans-serif'}
      position="relative"
      overflow="hidden"
    >
      <Starfield />
      <VStack
        spacing={8}
        p={12}
        maxW="800px"
        mx="auto"
        as={motion.div}
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition="0.8s ease-out"
      >
        <Logo />
        {user ? (
          <>             <Box
              as={motion.div}
              w="full"
              bg="rgba(255, 255, 255, 0.05)"
              backdropFilter="blur(10px)"
              border="1px solid rgba(0, 255, 255, 0.3)"
              borderRadius="lg"
              p={4}
              boxShadow="0 0 20px rgba(0, 255, 255, 0.2)"
              whileHover={{ boxShadow: '0 0 30px rgba(0, 255, 255, 0.4)' }}
            >
              <Input
                placeholder="Enter your cosmic query..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                bg="transparent"
                border="none"
                color="white"
                fontSize="lg"
                _focus={{ outline: 'none', borderBottom: '2px solid #00FFFF' }}
                _placeholder={{ color: 'rgba(255, 255, 255, 0.5)' }}
              />
            </Box>
            <Button
              onClick={handleGenerate}
              bg="linear-gradient(45deg, #00FFFF, #BB00FF)"
              color="white"
              p={6}
              borderRadius="lg"
              fontFamily={'"Orbitron", sans-serif'}
              fontWeight="bold"
              _hover={{
                bg: 'linear-gradient(45deg, #00CCCC, #AA00EE)',
                transform: 'translateY(-2px)',
                boxShadow: '0 0 20px rgba(0, 255, 255, 0.5)',
              }}
              isDisabled={loading}
              as={motion.div}
              whileTap={{ scale: 0.95 }}
            >
              {loading ? <Spinner size="sm" /> : 'Launch Query'}
            </Button>
            {response && (
              <Box
                bg="rgba(255, 255, 255, 0.05)"
                backdropFilter="blur(10px)"
                border="1px solid rgba(0, 255, 255, 0.3)"
                borderRadius="lg"
                p={6}
                maxW="600px"
                w="full"
                as={motion.div}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}n                transition="0.5s"
                boxShadow="0 0 20px rgba(0, 255, 255, 0.2)"
              >
                <Text fontSize="md" color="white">
                  {response}
                </Text>
              </Box>
            )}
          </>
        ) : (
          <Button
            onClick={handleLogin}
            bg="linear-gradient(45deg, #00FFFF, #BB00FF)"
            color="white"
            p={6}
            borderRadius="lg"
            fontFamily={'"Orbitron", sans-serif'}
            fontWeight="bold"
            _hover={{
              bg: 'linear-gradient(45deg, #00CCCC, #AA00EE)',
              transform: 'translateY(-2px)',
              boxShadow: '0 0 20px rgba(0, 255, 255, 0.5)',
            }}
            as={motion.div}
            whileTap={{ scale: 0.95 }}
          >
            Sign in with Google
          </Button>
        )}
      </VStack>
    </Box>
  );
}
