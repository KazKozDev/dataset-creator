import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Box,
  Flex,
  Text,
  VStack,
  HStack,
  Icon,
  Image,
  Heading,
  useColorModeValue,
  Divider,
  Link as ChakraLink,
} from '@chakra-ui/react';
import {
  FiHome,
  FiDatabase,
  FiSettings,
  FiCheckCircle,
  FiGrid,
  FiActivity,
  FiCopy,
  FiExternalLink,
} from 'react-icons/fi';

const NavItem = ({ icon, children, to, ...rest }) => {
  const location = useLocation();
  const isActive = location.pathname === to;
  const activeColor = useColorModeValue('blue.500', 'blue.300');
  const hoverBg = useColorModeValue('gray.100', 'gray.700');
  const activeStyle = isActive ? { color: activeColor, fontWeight: 'bold' } : {};

  return (
    <Link to={to} style={{ textDecoration: 'none', width: '100%' }}>
      <Flex
        align="center"
        p="4"
        role="group"
        cursor="pointer"
        fontWeight="medium"
        color={useColorModeValue('gray.600', 'gray.200')}
        _hover={{ bg: hoverBg, color: activeColor }}
        borderRadius="md"
        {...activeStyle}
        {...rest}
      >
        {icon && (
          <Icon
            ml="2"
            mr="1"
            fontSize="16"
            as={icon}
            color={isActive ? activeColor : 'inherit'}
          />
        )}
        {!icon && <Box ml="2" />}
        {children}
      </Flex>
    </Link>
  );
};

const Sidebar = () => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const textColor = useColorModeValue('gray.600', 'gray.200');

  return (
    <Box
      position="fixed"
      left={0}
      width="250px"
      height="100vh"
      bg={bgColor}
      borderRight="1px"
      borderRightColor={borderColor}
      display="flex"
      flexDirection="column"
    >
      {/* Logo/Title */}
      <Box p={6} borderBottomWidth="1px" borderColor={borderColor}>
        <HStack align="start" spacing={3}>
          <Image src="/logo.png" alt="Logo" boxSize="70px" objectFit="contain" />
          <VStack align="start" spacing={0}>
            <Heading size="md">Synthetic</Heading>
            <Heading size="md">Data</Heading>
            <Heading size="md">Foundry</Heading>
          </VStack>
        </HStack>
      </Box>

      <VStack spacing={0} align="stretch" flex="1">
        <NavItem icon={FiHome} to="/">
          Dashboard
        </NavItem>
        <NavItem icon={FiGrid} to="/generator">
          Generator
        </NavItem>
        <NavItem icon={FiCopy} to="/templates">
          Templates
        </NavItem>
        <NavItem icon={FiCheckCircle} to="/quality">
          Quality Control
        </NavItem>
        <NavItem icon={FiDatabase} to="/datasets">
          Datasets
        </NavItem>
        <NavItem icon={FiActivity} to="/tasks">
          Tasks
        </NavItem>

        {/* Settings at the bottom */}
        <Box px={4} py={3} mt={4}>
          <Divider />
        </Box>
        <NavItem icon={FiSettings} to="/settings">
          Settings
        </NavItem>
      </VStack>

      <Box p={4} mt="auto">
        <ChakraLink
          href="https://github.com/KazKozDev"
          color="black"
          fontSize="sm"
          display="inline-flex"
          alignItems="center"
          _hover={{ color: 'blue.500' }}
          mt={1}
          lineHeight="1.4"
          isExternal
        >
          KazKozDev
          <Icon as={FiExternalLink} ml={1} />
        </ChakraLink>
      </Box>
    </Box>
  );
};

export default Sidebar;