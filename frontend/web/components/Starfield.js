import { Box } from '@chakra-ui/react';
import { useEffect } from 'react';

export default function Starfield() {
  useEffect(() => {
    // Initialize starfield effect here or use a library
  }, []);

  return (
    <Box
      position="absolute"
      top={0}
      left={0}
      width="100%"
      height="100%"
      zIndex={-1}
      bg="black"
    >
      {/* Starfield canvas or effect */}
    </Box>
  );
}